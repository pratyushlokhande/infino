// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Moving a table's `ascii_lower` columns to the `standard` analyzer.
//!
//! Every other repair is per superfile, because every superfile can be
//! repaired without changing what the table's queries mean. This one cannot
//! be: queries tokenize once per table, with the analyzer the table's
//! options name, and the term index, the term-statistics sidecar and the
//! table-wide document frequencies all live in that one term space. A table
//! holding some superfiles of each analyzer answers wrongly for one of the
//! two halves. So every superfile is rebuilt first, without publishing any
//! of them, and then all of them are published in one manifest commit
//! together with the options that name the new analyzer.
//!
//! The rebuild holds no seal. A rebuild keeps every row in place, so an
//! input's tombstones describe its output exactly; they are read under a
//! seal at publish time and carried across. Deletes keep landing on the
//! inputs for the whole length of the rebuild, which on a large table is
//! hours.

use std::{collections::HashMap, slice, sync::Arc, time::Duration};

use uuid::Uuid;

use crate::{
    config::ReindexMode,
    supertable::{
        Supertable, SupertableOptions,
        error::{CompactionError, ReindexError},
        manifest::{SuperfileEntry, listed_once},
        optimize::compact::{BatchCommit, SuperfileMerge},
        reindex::{PlannedRepair, ReindexReport, Repair, build::RepairMerge},
        writer::{PreparedSuperfile, write_superfile_list},
    },
};

/// Rounds a run spends catching up with a table that keeps changing under
/// it before giving up.
///
/// A round is one pass that found superfiles it had not rebuilt (another
/// writer appended, or compacted what this run had rebuilt), or one publish
/// that lost its inputs to another writer. Each costs a rebuild of only what
/// changed, so a few are cheap. A table under constant ingest never runs
/// out of them, and the error then says to pause ingest.
const MAX_CATCH_UP_ROUNDS: usize = 3;

impl Supertable {
    /// The superfiles an analyzer change rebuilds: every live one, while
    /// the table has an `ascii_lower` column, and none after.
    pub(super) fn plan_standard_analyzer(&self) -> Vec<PlannedRepair> {
        let manifest = self.inner().manifest.load_full();
        if manifest.options.ascii_lower_columns().next().is_none() {
            return Vec::new();
        }
        let mut planned: Vec<PlannedRepair> =
            listed_once(manifest.get_all_superfiles(), |e| e.superfile_id)
                .map(|e| PlannedRepair {
                    superfile_id: e.superfile_id,
                    mode: ReindexMode::ToStandardAnalyzer,
                    live_bytes: e.subsection_offsets.as_ref().map_or(0, |o| o.total_size),
                })
                .collect();
        planned.sort_by_key(|p| p.superfile_id);
        planned
    }

    /// Rebuild every superfile under `standard` and publish them together
    /// with the new analyzer; see the module docs.
    pub(super) async fn change_to_standard_analyzer(
        &self,
        stale_seal_timeout: Duration,
    ) -> Result<ReindexReport, ReindexError> {
        let manifest = self.inner().manifest.load_full();
        let total = manifest.get_all_superfiles().len();
        if manifest.options.ascii_lower_columns().next().is_none() {
            return Ok(ReindexReport {
                already_current: total,
                ..Default::default()
            });
        }
        let index_only: Vec<String> = manifest
            .options
            .ascii_lower_columns()
            .filter(|c| !c.stored)
            .map(|c| c.column.clone())
            .collect();
        if !index_only.is_empty() {
            return Err(ReindexError::IndexOnlyColumns {
                columns: index_only,
            });
        }

        // Shared with compaction for the same reason a reindex shares it: a
        // compaction here would replace superfiles this run has rebuilt.
        let _slot = self
            .try_hold_compaction_slot()
            .ok_or(ReindexError::AlreadyRunning)?;

        let target = Arc::new(manifest.options.with_standard_analyzer());
        let merge: Arc<dyn SuperfileMerge> = Arc::new(RepairMerge::new(Repair::Standard, false));
        // Input superfile → its rebuilt, uploaded replacement.
        let mut rebuilt: HashMap<Uuid, Arc<SuperfileEntry>> = HashMap::new();
        let mut rounds = 0;
        loop {
            self.refresh()
                .await
                .map_err(|e| ReindexError::Publish(e.to_string()))?;
            let current = self.inner().manifest.load_full();
            let live: Vec<Arc<SuperfileEntry>> =
                listed_once(current.get_all_superfiles().iter().cloned(), |e| {
                    e.superfile_id
                })
                .collect();
            // A rebuilt input another writer has since replaced is gone;
            // its upload is an orphan for gc.
            rebuilt.retain(|id, _| live.iter().any(|e| e.superfile_id == *id));

            let mut pairs: Vec<(Arc<SuperfileEntry>, Arc<SuperfileEntry>)> =
                Vec::with_capacity(live.len());
            let mut missing: Vec<&Arc<SuperfileEntry>> = Vec::new();
            for input in &live {
                match rebuilt.get(&input.superfile_id) {
                    Some(output) => pairs.push((Arc::clone(input), Arc::clone(output))),
                    None => missing.push(input),
                }
            }
            if !missing.is_empty() {
                if !rebuilt.is_empty() {
                    rounds += 1;
                    if rounds > MAX_CATCH_UP_ROUNDS {
                        return Err(ReindexError::TableKeptChanging { rounds });
                    }
                }
                for input in missing {
                    let output = self.rebuild_and_upload(input, &merge).await?;
                    rebuilt.insert(input.superfile_id, output);
                }
                // Something may have landed while this round rebuilt.
                continue;
            }

            // An append or update in flight on this handle built under
            // `ascii_lower`; the publish bumps the handle's options
            // generation, which refuses that commit.
            match self
                .publish_rebuilt(&pairs, &merge, &target, stale_seal_timeout)
                .await
            {
                Ok(()) => {
                    // The commit carried no term-index contributions, so the
                    // index is marked incomplete: rebuild it over the new
                    // superfiles, whose terms the old index never held.
                    self.refresh_term_stats_best_effort();
                    return Ok(ReindexReport {
                        rewritten: pairs.len(),
                        ..Default::default()
                    });
                }
                Err(e) if lost_to_another_writer(&e) => {
                    rounds += 1;
                    if rounds > MAX_CATCH_UP_ROUNDS {
                        return Err(ReindexError::TableKeptChanging { rounds });
                    }
                }
                Err(e) => return Err(ReindexError::Publish(e.to_string())),
            }
        }
    }

    /// Rebuild `input` under `merge` and put the result in storage, without
    /// publishing it.
    ///
    /// The bytes are dropped once they land, so a run over a large table
    /// holds one superfile at a time rather than all of them until the
    /// commit.
    async fn rebuild_and_upload(
        &self,
        input: &Arc<SuperfileEntry>,
        merge: &Arc<dyn SuperfileMerge>,
    ) -> Result<Arc<SuperfileEntry>, ReindexError> {
        let rewrite = |cause: String| ReindexError::Rewrite {
            superfile_id: input.superfile_id,
            cause,
        };
        // No tombstones: the rebuild keeps every row, deleted ones too, and
        // the deletes are carried at publish time.
        let PreparedSuperfile {
            entry,
            bytes_for_storage,
            ..
        } = self
            .merge_superfiles_with(slice::from_ref(input), &HashMap::new(), Arc::clone(merge))
            .await
            .map_err(|e| rewrite(e.to_string()))?;
        let write =
            bytes_for_storage.ok_or_else(|| rewrite("the rebuild wrote no bytes".into()))?;
        let options = &self.inner().options;
        let storage = options.storage.as_ref().ok_or(ReindexError::NoStorage)?;
        write_superfile_list(storage, options, &mut vec![write], &mut Vec::new())
            .await
            .map_err(|e| rewrite(e.to_string()))?;
        Ok(entry)
    }

    /// Seal every input, carry its tombstones onto its rebuild, and publish
    /// all of them with `target` in one manifest commit, or none of them.
    async fn publish_rebuilt(
        &self,
        pairs: &[(Arc<SuperfileEntry>, Arc<SuperfileEntry>)],
        merge: &Arc<dyn SuperfileMerge>,
        target: &Arc<SupertableOptions>,
        stale_seal_timeout: Duration,
    ) -> Result<(), CompactionError> {
        let batch = self
            .prepare_uploaded_batch(pairs, merge.as_ref(), stale_seal_timeout)
            .await?;
        self.commit_compaction_batch(batch, &BatchCommit::WholeTable(Arc::clone(target)))
            .await
    }
}

/// Whether a publish failed because another writer moved the table under
/// it, which another round can absorb, rather than for a reason that will
/// repeat.
fn lost_to_another_writer(e: &CompactionError) -> bool {
    matches!(
        e,
        CompactionError::SuperfileNotFound(_)
            | CompactionError::UnplannedSuperfile(_)
            | CompactionError::SidecarChangedUnderSeal { .. }
            | CompactionError::SidecarConflict { .. }
            | CompactionError::SealRetriesExhausted { .. }
    )
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{builder::FtsConfig, fts::tokenize::ASCII_LOWER_TOKENIZER},
        supertable::{
            error::CommitError,
            writer::{CommitListMetadata, persist_commit_async},
        },
        test_helpers::{build_title_batch, default_supertable_options},
    };

    /// Long enough that no seal this test places can go stale under it.
    const SEAL_TIMEOUT: Duration = Duration::from_secs(600);

    /// A storage-backed table whose one full-text column is `ascii_lower`.
    fn ascii_lower_table(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut options = default_supertable_options().with_storage(storage);
        options.fts_columns = vec![FtsConfig::new("title").analyzer(ASCII_LOWER_TOKENIZER)];
        Supertable::create(options).expect("create")
    }

    fn commit_titles(table: &Supertable, titles: &[&str]) {
        let mut w = table.writer().expect("writer");
        w.append(&build_title_batch(titles)).expect("append");
        w.commit().expect("commit");
    }

    fn superfile_ids(table: &Supertable) -> Vec<Uuid> {
        let mut ids: Vec<Uuid> = table
            .inner()
            .manifest
            .load()
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        ids.sort();
        ids
    }

    /// A publish that would leave a superfile it did not rebuild refuses
    /// outright: that superfile holds `ascii_lower` terms, and the table
    /// would be stamped `standard` over it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_publish_refuses_a_superfile_it_does_not_replace() {
        let dir = TempDir::new().expect("tempdir");
        let table = ascii_lower_table(&dir);
        commit_titles(&table, &["don't panic"]);
        commit_titles(&table, &["café crème"]);
        let entries = table.inner().manifest.load().get_all_superfiles().to_vec();
        let [first, second] = entries.as_slice() else {
            panic!("expected two superfiles, got {}", entries.len());
        };
        let ids_before = superfile_ids(&table);

        let merge: Arc<dyn SuperfileMerge> = Arc::new(RepairMerge::new(Repair::Standard, false));
        let target = Arc::new(table.inner().options.with_standard_analyzer());
        let first_rebuilt = table
            .rebuild_and_upload(first, &merge)
            .await
            .expect("rebuild the first");

        let err = table
            .publish_rebuilt(
                &[(Arc::clone(first), Arc::clone(&first_rebuilt))],
                &merge,
                &target,
                SEAL_TIMEOUT,
            )
            .await
            .expect_err("half a table must not publish");
        assert!(
            matches!(err, CompactionError::UnplannedSuperfile(id) if id == second.superfile_id),
            "{err}"
        );
        assert_eq!(superfile_ids(&table), ids_before, "nothing was published");
        assert!(
            table
                .inner()
                .manifest
                .load()
                .options
                .ascii_lower_columns()
                .next()
                .is_some(),
            "the analyzer did not move"
        );

        // Publishing both succeeds, which it could not if the refused
        // attempt had left its seal on the first superfile.
        let second_rebuilt = table
            .rebuild_and_upload(second, &merge)
            .await
            .expect("rebuild the second");
        table
            .publish_rebuilt(
                &[
                    (Arc::clone(first), first_rebuilt),
                    (Arc::clone(second), second_rebuilt),
                ],
                &merge,
                &target,
                SEAL_TIMEOUT,
            )
            .await
            .expect("publish the whole table");
        let manifest = table.inner().manifest.load();
        assert!(manifest.options.ascii_lower_columns().next().is_none());
        assert_eq!(manifest.get_all_superfiles().len(), 2);
        assert!(
            manifest
                .get_all_superfiles()
                .iter()
                .all(|e| !ids_before.contains(&e.superfile_id)),
            "both superfiles were replaced"
        );
    }

    /// What a write built under the options before a change is refused at
    /// publish: the change moved the handle's options generation, and the
    /// publish carries the one it built under.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_publish_built_before_the_change_is_refused() {
        let dir = TempDir::new().expect("tempdir");
        let table = ascii_lower_table(&dir);
        commit_titles(&table, &["don't panic"]);
        let built_under = table.inner().options_generation();

        table
            .change_to_standard_analyzer(SEAL_TIMEOUT)
            .await
            .expect("move to standard");
        assert_ne!(
            table.inner().options_generation(),
            built_under,
            "publishing new options moves the generation"
        );

        let storage = table
            .inner()
            .options
            .storage
            .clone()
            .expect("storage-backed");
        let refused = persist_commit_async(
            table.inner(),
            storage,
            Vec::new(),
            &[],
            Vec::new(),
            Vec::new(),
            CommitListMetadata::empty(),
            Vec::new(),
            Some(built_under),
        )
        .await
        .expect_err("a build from before the change must not publish");
        assert!(matches!(refused, CommitError::OptionsChanged), "{refused}");
        assert!(refused.is_conflict(), "refused retryably");
    }
}
