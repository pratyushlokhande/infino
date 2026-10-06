// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Moving an `ascii_lower` table to the `standard` analyzer.
//!
//! The inputs are real tables: one written by a current release on an
//! explicit `ascii_lower` analyzer, and one written by a release where
//! `ascii_lower` was the default. The end state asserted is the same for
//! both: every column on `standard`, every superfile in the current format
//! with its analysis recorded, and the table answering as a `standard` table
//! does, with its rows, deletes, filters and vectors as they were.

use std::{fs, path::Path, sync::Arc, time::Duration};

use datafusion::prelude::{col, lit};
use infino::{
    BoolMode, CompactionSettings, ConnectOptions, Connection, Consistency, InfinoError,
    OptimizeOptions, ReindexError, ReindexMode, ReindexOptions, Supertable,
    arrow_array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray,
        RecordBatch,
    },
    arrow_schema::{DataType, Field},
    connect, connect_with,
    superfile::format::fts::VERSION_CURRENT,
};

use crate::corpus_shapes::{
    ANALYZER_PROBES, ASCII_LOWER_SHAPE_LIVE_ROWS, EMBEDDING_DIM, N_DOCS, TABLE, TITLES_WITH_JUMP,
    blob_versions, connect_corpus, corpus_dir, embedding, fts_column, fts_columns, hits, hits_k,
    open_corpus, probe_embedding, superfile_paths, vector_hits,
};

/// Shared-corpus documents whose `notes` column carries text.
const DOCS_WITH_NOTES: usize = N_DOCS as usize / 4;

/// Shared-corpus documents whose `body` holds both `common` and `shared`:
/// three in every four.
const SHARED_DOCS: u64 = N_DOCS as u64 / 4 * 3;

/// Shared-corpus titles reading `quick brown fox`: one in every three.
const QUICK_BROWN_TITLES: usize = N_DOCS as usize / 3;

/// A word only `standard` indexes, appended after the migration to show the
/// handle that ran it now builds under `standard`.
const POST_MIGRATION_WORD: &str = "émigré";

/// The analyzer every column of `root`'s superfiles names, per superfile.
fn column_analyzers(root: &Path, column: &str) -> Vec<String> {
    fts_columns(root)
        .iter()
        .map(|columns| {
            fts_column(columns, column)["tokenizer"]
                .as_str()
                .expect("tokenizer is a string")
                .to_string()
        })
        .collect()
}

/// The analyzer names the catalog records for the corpus table.
fn catalog_analyzers(root: &Path) -> Vec<String> {
    let body: serde_json::Value = serde_json::from_slice(
        &fs::read(root.join("_catalog").join("current")).expect("read catalog"),
    )
    .expect("catalog is JSON");
    body["tables"][TABLE]["fts_analyzers"]
        .as_array()
        .expect("analyzer list")
        .iter()
        .map(|a| a.as_str().expect("analyzer name").to_string())
        .collect()
}

/// Overwrite the analyzer names the catalog records for the corpus table,
/// as a run that stopped before its catalog write would have left them.
fn set_catalog_analyzers(root: &Path, analyzers: &[&str]) {
    let path = root.join("_catalog").join("current");
    let mut body: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).expect("read catalog")).expect("catalog is JSON");
    body["tables"][TABLE]["fts_analyzers"] = serde_json::json!(analyzers);
    fs::write(&path, serde_json::to_vec(&body).expect("encode catalog")).expect("write catalog");
}

/// The distances of a vector search's hits, in rank order.
fn distances(hits: &[(i128, f32)]) -> Vec<f32> {
    hits.iter().map(|(_, distance)| *distance).collect()
}

/// One row for the `v7_ascii_lower` schema whose `body` is `text`.
fn one_row(table: &Supertable, text: &str) -> RecordBatch {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(vec![text])),
        Arc::new(LargeStringArray::from(vec!["appended"])),
        Arc::new(LargeStringArray::from(vec![None::<&str>])),
        Arc::new(
            FixedSizeListArray::try_new(
                item,
                EMBEDDING_DIM as i32,
                Arc::new(Float32Array::from(probe_embedding())),
                None,
            )
            .expect("embedding"),
        ),
    ];
    RecordBatch::try_new(table.schema(), columns).expect("row matches schema")
}

/// The whole round trip on the table a current release writes.
#[test]
fn moves_an_ascii_lower_table_to_standard_in_the_current_format() {
    const SHAPE: &str = "v7_ascii_lower";
    let Some((_tmp, table, root)) = open_corpus(SHAPE) else {
        return;
    };
    let k = ASCII_LOWER_SHAPE_LIVE_ROWS;
    let superfiles = superfile_paths(&root).len();
    let vectors_before = vector_hits(&table, &probe_embedding());

    let before = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess");
    assert_eq!(before.ascii_lower_columns, ["body", "title"]);
    // The analyzer is the only thing behind: a plain reindex has nothing to
    // do, which is why it takes a mode of its own.
    assert!(before.is_current(), "{before:?}");

    let plan = table
        .reindex_plan(&ReindexOptions::to_standard_analyzer())
        .expect("plan");
    assert_eq!(plan.len(), superfiles, "every superfile is rebuilt");
    assert!(
        plan.iter()
            .all(|p| p.mode == ReindexMode::ToStandardAnalyzer)
    );

    let report = table
        .reindex(&ReindexOptions::to_standard_analyzer())
        .expect("move to standard");
    assert_eq!(report.rewritten, superfiles);

    // The files: every column on `standard`, the current container, the
    // analysis recorded, and the filters `title` was created with kept.
    table
        .gc(Duration::ZERO)
        .expect("collect replaced superfiles");
    assert!(
        blob_versions(&root).iter().all(|v| *v == VERSION_CURRENT),
        "{:?}",
        blob_versions(&root)
    );
    for column in ["body", "title", "notes"] {
        assert!(
            column_analyzers(&root, column)
                .iter()
                .all(|a| a == "standard"),
            "{column}: {:?}",
            column_analyzers(&root, column)
        );
    }
    for columns in fts_columns(&root) {
        let title = fts_column(&columns, "title");
        assert_eq!(title["stopwords"], "english");
        assert_eq!(title["stemmer"], "english");
        assert_eq!(title["positions"], true);
        assert!(title["analysis_revision"].is_u64(), "{title}");
    }

    // The answers: every probe at its `standard` count, the deleted row
    // still deleted, and the columns and vectors that were not moved
    // unchanged.
    for probe in ANALYZER_PROBES {
        assert_eq!(
            hits_k(&table, "body", probe.term, k),
            probe.standard_hits,
            "body {:?}",
            probe.term
        );
    }
    assert_eq!(hits_k(&table, "body", "common", k), N_DOCS as usize);
    assert_eq!(hits_k(&table, "title", "jumping", k), TITLES_WITH_JUMP);
    assert_eq!(hits_k(&table, "notes", "sparse", k), DOCS_WITH_NOTES);
    // Compared by distance, not id: the corpus embeds whole groups of
    // documents identically, so the probe's neighbours tie and which of them
    // fill the top k follows superfile order, which the change renumbers.
    // The vector bytes are carried, so every distance must be unchanged.
    assert_eq!(
        distances(&vector_hits(&table, &probe_embedding())),
        distances(&vectors_before)
    );

    let after = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess after");
    assert!(after.is_current(), "{after:?}");
    assert!(after.ascii_lower_columns.is_empty(), "{after:?}");

    // Idempotent: nothing left on `ascii_lower`, so nothing to rebuild.
    let again = table
        .reindex(&ReindexOptions::to_standard_analyzer())
        .expect("second run");
    assert_eq!(again.rewritten, 0);
    assert!(
        table
            .reindex_plan(&ReindexOptions::to_standard_analyzer())
            .expect("plan after")
            .is_empty()
    );

    // The handle that ran it now writes `standard` superfiles.
    table
        .append(&one_row(&table, POST_MIGRATION_WORD))
        .expect("append after the change");
    assert_eq!(hits_k(&table, "body", POST_MIGRATION_WORD, k + 1), 1);
}

/// The change records `standard` in the catalog in the same operation, so
/// an engine that builds the table's options from its record opens it as
/// it now is without this engine having opened it first.
#[test]
fn the_change_records_standard_in_the_catalog() {
    const SHAPE: &str = "v7_ascii_lower";
    let Some((_tmp, db, root)) = connect_corpus(SHAPE) else {
        return;
    };
    assert_eq!(
        catalog_analyzers(&root),
        ["ascii_lower", "ascii_lower", "standard"]
    );
    db.open_table(TABLE)
        .expect("open")
        .reindex(&ReindexOptions::to_standard_analyzer())
        .expect("move to standard");
    assert_eq!(
        catalog_analyzers(&root),
        ["standard", "standard", "standard"]
    );
}

/// A run that stopped after its manifest commit and before its catalog
/// write leaves the record naming `ascii_lower`; the next open still finds
/// the table on `standard`, and corrects the record.
#[test]
fn an_open_corrects_a_record_the_change_did_not_reach() {
    const SHAPE: &str = "v7_ascii_lower";
    let Some((_tmp, db, root)) = connect_corpus(SHAPE) else {
        return;
    };
    db.open_table(TABLE)
        .expect("open")
        .reindex(&ReindexOptions::to_standard_analyzer())
        .expect("move to standard");
    drop(db);
    set_catalog_analyzers(&root, &["ascii_lower", "ascii_lower", "standard"]);

    let db = connect(root.to_str().expect("utf-8 path")).expect("reconnect");
    let table = db.open_table(TABLE).expect("open a migrated table");
    for probe in ANALYZER_PROBES {
        assert_eq!(
            hits_k(&table, "body", probe.term, ASCII_LOWER_SHAPE_LIVE_ROWS),
            probe.standard_hits,
            "body {:?}",
            probe.term
        );
    }
    assert_eq!(
        catalog_analyzers(&root),
        ["standard", "standard", "standard"]
    );
}

/// Handles opened before the change move to it rather than failing: a
/// reader answers under `standard` once it refreshes, and a writer's
/// appends land under `standard` — at most one is refused, retryably, for
/// having been built under the old analyzer.
#[test]
fn a_handle_opened_before_the_change_moves_to_it() {
    const SHAPE: &str = "v7_ascii_lower";
    let Some((_tmp, _db, root)) = connect_corpus(SHAPE) else {
        return;
    };
    let path = root.to_str().expect("utf-8 path");
    // Strong, so every query refreshes and the reader's move is observable
    // without waiting out a staleness window.
    let reader = connect_with(
        path,
        ConnectOptions::new().with_read_consistency(Consistency::Strong),
    )
    .expect("reader connection")
    .open_table(TABLE)
    .expect("open the reader before the change");
    let writer = connect(path)
        .expect("writer connection")
        .open_table(TABLE)
        .expect("open the writer before the change");
    let k = ASCII_LOWER_SHAPE_LIVE_ROWS;
    assert_eq!(
        hits_k(&reader, "body", "résumé", k),
        0,
        "ascii_lower before"
    );

    connect(path)
        .expect("migrating connection")
        .open_table(TABLE)
        .expect("open")
        .reindex(&ReindexOptions::to_standard_analyzer())
        .expect("move to standard");

    assert_eq!(hits_k(&reader, "body", "résumé", k), 1, "standard after");
    assert_eq!(hits_k(&reader, "body", "don", k), 0, "standard after");

    let row = one_row(&writer, POST_MIGRATION_WORD);
    if let Err(refused) = writer.append(&row) {
        assert!(matches!(refused, InfinoError::Conflict(_)), "{refused}");
        writer
            .append(&row)
            .expect("the retry builds under standard");
    }
    assert_eq!(hits_k(&writer, "body", POST_MIGRATION_WORD, k + 1), 1);
    assert_eq!(hits_k(&reader, "body", POST_MIGRATION_WORD, k + 1), 1);

    // What matters most: nothing built under `ascii_lower` landed.
    writer
        .gc(Duration::ZERO)
        .expect("collect replaced superfiles");
    assert!(
        column_analyzers(&root, "body")
            .iter()
            .all(|a| a == "standard"),
        "{:?}",
        column_analyzers(&root, "body")
    );
}

/// An index-only `ascii_lower` column has no text to re-analyze, so the run
/// refuses before writing anything.
#[test]
fn refuses_a_table_with_an_index_only_ascii_lower_column() {
    const SHAPE: &str = "v7_ascii_lower_index_only";
    let Some((_tmp, table, root)) = open_corpus(SHAPE) else {
        return;
    };
    let files_before = superfile_paths(&root);

    let err = table
        .reindex(&ReindexOptions::to_standard_analyzer())
        .expect_err("an index-only column cannot be re-analyzed");
    match err {
        ReindexError::IndexOnlyColumns { columns } => assert_eq!(columns, ["body"]),
        other => panic!("expected IndexOnlyColumns, got {other}"),
    }

    assert_eq!(superfile_paths(&root), files_before, "nothing was written");
    for probe in ANALYZER_PROBES {
        assert_eq!(
            hits_k(&table, "body", probe.term, ASCII_LOWER_SHAPE_LIVE_ROWS),
            probe.ascii_lower_hits,
            "body {:?}",
            probe.term
        );
    }
}

/// A table from a release where `ascii_lower` was the default, in an old
/// container: one run brings it to `standard` and the current format.
#[test]
fn moves_a_table_written_on_the_old_default() {
    const SHAPE: &str = "v4_bitset_blocks";
    let Some(_) = corpus_dir(SHAPE) else {
        return;
    };
    let (_tmp, table, root) = open_corpus(SHAPE).expect("corpus present");
    assert_eq!(
        table
            .index_staleness(&ReindexOptions::default())
            .expect("assess")
            .ascii_lower_columns,
        ["body", "title", "notes"]
    );
    assert_eq!(hits(&table, "body", "\u{1f525}"), 0);

    table
        .reindex(&ReindexOptions::to_standard_analyzer())
        .expect("move to standard");
    table
        .gc(Duration::ZERO)
        .expect("collect replaced superfiles");

    assert!(
        blob_versions(&root).iter().all(|v| *v == VERSION_CURRENT),
        "{:?}",
        blob_versions(&root)
    );
    assert!(
        column_analyzers(&root, "body")
            .iter()
            .all(|a| a == "standard")
    );
    // Terms only `standard` keeps are reachable now.
    assert_eq!(hits(&table, "body", "\u{1f525}"), 1);
    assert_eq!(hits(&table, "body", "café"), 1);
    assert_eq!(hits(&table, "body", "common"), N_DOCS as usize);
    let after = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess after");
    assert!(
        after.is_current() && after.ascii_lower_columns.is_empty(),
        "{after:?}"
    );
}

/// A table already on `standard` has nothing to move.
#[test]
fn leaves_a_standard_table_alone() {
    const SHAPE: &str = "v6_positional";
    let Some((_tmp, table, root)) = open_corpus(SHAPE) else {
        return;
    };
    let files_before = superfile_paths(&root);
    assert!(
        table
            .reindex_plan(&ReindexOptions::to_standard_analyzer())
            .expect("plan")
            .is_empty()
    );
    let report = table
        .reindex(&ReindexOptions::to_standard_analyzer())
        .expect("no-op run");
    assert_eq!(report.rewritten, 0);
    assert_eq!(report.already_current, files_before.len());
    assert_eq!(superfile_paths(&root), files_before);
}

/// Rows a fresh probe embedding has no tie with: the corpus puts its weight
/// on one dimension per document, these on two.
fn unique_embedding(first: usize) -> Vec<f32> {
    (0..EMBEDDING_DIM)
        .map(|d| match d == first || d == first + 1 {
            true => 1.0,
            false => 0.0,
        })
        .collect()
}

/// One row with every column chosen, for the post-migration appends.
fn row(table: &Supertable, body: &str, title: &str, embedding: Vec<f32>) -> RecordBatch {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(vec![body])),
        Arc::new(LargeStringArray::from(vec![title])),
        Arc::new(LargeStringArray::from(vec![None::<&str>])),
        Arc::new(
            FixedSizeListArray::try_new(
                item,
                EMBEDDING_DIM as i32,
                Arc::new(Float32Array::from(embedding)),
                None,
            )
            .expect("embedding"),
        ),
    ];
    RecordBatch::try_new(table.schema(), columns).expect("row matches schema")
}

/// The `body` of every row `batches` returns, in order.
fn bodies(batches: &[RecordBatch]) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            let column = b
                .column_by_name("body")
                .expect("body projected")
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("body is LargeUtf8");
            (0..column.len())
                .map(|i| column.value(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Rows a table holds, through SQL.
fn sql_rows(db: &Connection) -> i64 {
    let batches = db
        .query_sql(&format!("SELECT count(*) AS n FROM {TABLE}"))
        .expect("count through SQL");
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count is Int64")
        .value(0)
}

/// After the change, every read path answers and every write path works:
/// full-text search in each of its forms, vector and hybrid search, appends
/// whose text only `standard` indexes, a delete and an update of those rows,
/// a compaction over the migrated superfiles, and a reopen through the
/// catalog.
#[test]
fn a_migrated_table_reads_and_writes_on_every_path() {
    const SHAPE: &str = "v7_ascii_lower";
    let Some((_tmp, db, root)) = connect_corpus(SHAPE) else {
        return;
    };
    let table = db.open_table(TABLE).expect("open");
    // Every embedding group, so each one's neighbours are checked rather
    // than only the first's.
    let groups: Vec<Vec<f32>> = (0..EMBEDDING_DIM as u32).map(embedding).collect();
    let neighbours_before: Vec<Vec<f32>> = groups
        .iter()
        .map(|g| distances(&vector_hits(&table, g)))
        .collect();

    table
        .reindex(&ReindexOptions::to_standard_analyzer())
        .expect("move to standard");
    let mut live = ASCII_LOWER_SHAPE_LIVE_ROWS;

    // Full-text search, every form.
    assert_eq!(sql_rows(&db), live as i64);
    assert_eq!(hits_k(&table, "body", "common", live), N_DOCS as usize);
    assert_eq!(hits_k(&table, "body", "don't", live), 1, "a standard term");
    assert_eq!(hits_k(&table, "body", "3.14", live), 1, "a standard term");
    assert_eq!(
        table
            .count("body", "common shared", BoolMode::And)
            .expect("count"),
        SHARED_DOCS
    );
    assert_eq!(
        hits_k(&table, "title", "\"quick brown\"", live),
        QUICK_BROWN_TITLES,
        "a phrase over the positional column"
    );
    assert_eq!(
        hits_k(&table, "title", "jumping", live),
        TITLES_WITH_JUMP,
        "the stemmer"
    );
    let exact = table
        .exact_match("body", "résumé attached", Some(&["body"]))
        .expect("exact match");
    assert_eq!(bodies(&exact), ["résumé attached"]);

    // Vector search: every group's neighbours at the distances they had.
    for (group, before) in groups.iter().zip(&neighbours_before) {
        let after = distances(&vector_hits(&table, group));
        assert!(!after.is_empty(), "a group returned no neighbours");
        assert_eq!(&after, before);
    }

    // Appends after the change, with text only `standard` indexes and
    // embeddings nothing in the corpus ties with.
    let rocket = unique_embedding(0);
    let tokyo = unique_embedding(2);
    table
        .append(&row(
            &table,
            "naïve rocket 🛸 don't",
            "fresh arrival",
            rocket.clone(),
        ))
        .expect("append the first row");
    table
        .append(&row(
            &table,
            "東京 tower résumé",
            "second arrival",
            tokyo.clone(),
        ))
        .expect("append the second row");
    live += 2;
    assert_eq!(sql_rows(&db), live as i64);
    assert_eq!(hits_k(&table, "body", "rocket", live), 1);
    assert_eq!(hits_k(&table, "body", "🛸", live), 1);
    assert_eq!(hits_k(&table, "body", "don't", live), 2);
    assert_eq!(hits_k(&table, "body", "résumé", live), 2);
    assert_eq!(
        hits_k(&table, "body", "naïve", live),
        2,
        "the shared corpus's non-ASCII document and the appended one"
    );
    let nearest = table
        .vector_search("emb", &rocket, 1, None, Some(&["body"]))
        .expect("vector search");
    assert_eq!(bodies(&nearest), ["naïve rocket 🛸 don't"]);
    let nearest = table
        .vector_search("emb", &tokyo, 1, None, Some(&["body"]))
        .expect("vector search");
    assert_eq!(bodies(&nearest), ["東京 tower résumé"]);
    let hybrid = table
        .hybrid_search(
            "body",
            "rocket",
            BoolMode::Or,
            "emb",
            &rocket,
            1,
            Some(&["body"]),
        )
        .expect("hybrid search");
    assert_eq!(bodies(&hybrid), ["naïve rocket 🛸 don't"]);

    // A delete and an update of rows written after the change.
    let deleted = table
        .delete(col("title").eq(lit("fresh arrival")))
        .expect("delete");
    assert_eq!(deleted.n_tombstoned(), 1);
    live -= 1;
    assert_eq!(hits_k(&table, "body", "rocket", live), 0);
    assert_eq!(hits_k(&table, "body", "🛸", live), 0);
    let nearest = table
        .vector_search("emb", &rocket, 1, None, Some(&["body"]))
        .expect("vector search");
    assert_ne!(
        bodies(&nearest),
        ["naïve rocket 🛸 don't"],
        "deleted row returned"
    );
    table
        .update(
            col("title").eq(lit("second arrival")),
            &row(
                &table,
                "東京 tower café crème",
                "second arrival",
                tokyo.clone(),
            ),
        )
        .expect("update");
    assert_eq!(sql_rows(&db), live as i64);
    assert_eq!(hits_k(&table, "body", "crème", live), 1);
    assert_eq!(
        hits_k(&table, "body", "résumé", live),
        1,
        "the updated row's old text is gone"
    );

    // Compaction over the migrated superfiles keeps every answer.
    table
        .optimize(&OptimizeOptions::compact(CompactionSettings {
            target_superfile_size_mb: 1,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        }))
        .expect("optimize");
    assert_eq!(sql_rows(&db), live as i64);
    assert_eq!(hits_k(&table, "body", "common", live), N_DOCS as usize);
    assert_eq!(hits_k(&table, "body", "crème", live), 1);
    assert_eq!(
        hits_k(&table, "title", "\"quick brown\"", live),
        QUICK_BROWN_TITLES
    );
    let nearest = table
        .vector_search("emb", &tokyo, 1, None, Some(&["body"]))
        .expect("vector search");
    assert_eq!(bodies(&nearest), ["東京 tower café crème"]);

    // A reopen through the catalog sees all of it, on `standard`, current.
    drop(table);
    drop(db);
    let db = connect(root.to_str().expect("utf-8 path")).expect("reconnect");
    let table = db.open_table(TABLE).expect("reopen");
    assert_eq!(sql_rows(&db), live as i64);
    assert_eq!(hits_k(&table, "body", "crème", live), 1);
    assert_eq!(hits_k(&table, "body", "don't", live), 1);
    for (group, before) in groups.iter().zip(&neighbours_before) {
        assert_eq!(&distances(&vector_hits(&table, group)), before);
    }
    let nearest = table
        .vector_search("emb", &tokyo, 1, None, Some(&["body"]))
        .expect("vector search");
    assert_eq!(bodies(&nearest), ["東京 tower café crème"]);
    let staleness = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess");
    assert!(
        staleness.is_current() && staleness.ascii_lower_columns.is_empty(),
        "{staleness:?}"
    );
    table.gc(Duration::ZERO).expect("collect");
    assert!(
        blob_versions(&root).iter().all(|v| *v == VERSION_CURRENT),
        "{:?}",
        blob_versions(&root)
    );
    assert!(
        column_analyzers(&root, "body")
            .iter()
            .all(|a| a == "standard")
    );
}
