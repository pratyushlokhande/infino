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

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use bytes::Bytes;
use datafusion::prelude::{col, lit};
use infino::{
    BoolMode, CompactionSettings, ConnectOptions, Connection, Consistency, InfinoError,
    OptimizeOptions, ReindexError, ReindexMode, ReindexOptions, Supertable,
    arrow_array::{
        Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, Int64Array,
        LargeStringArray, RecordBatch,
    },
    arrow_schema::{DataType, Field},
    connect, connect_with,
    superfile::{
        SuperfileReader,
        format::{fts::VERSION_CURRENT, kv},
    },
    supertable::manifest::SuperfileEntry,
};
use uuid::Uuid;

use crate::corpus_shapes::{
    ANALYZER_PROBES, ASCII_LOWER_SHAPE_LIVE_ROWS, EMBEDDING_DIM, N_DOCS, TABLE, TITLES_WITH_JUMP,
    blob_versions, connect_corpus, corpus_dir, embedding, files_with_extension, first_region,
    fts_column, fts_columns, hits, hits_k, open_corpus, probe_embedding, raw_footer_kvs,
    superfile_paths, table_dir, vector_hits,
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

/// What the audit compares about one superfile.
struct SuperfileFacts {
    superfile_id: Uuid,
    bytes: Bytes,
    fts: (u64, u64),
    vec: Option<(u64, u64)>,
    ids: Vec<i128>,
    columns: Vec<serde_json::Value>,
    birth_version: u64,
    partition_key: Vec<u8>,
}

/// Read `entry`'s superfile under `dir` and the facts the audit needs.
fn superfile_facts(dir: &Path, entry: &SuperfileEntry) -> SuperfileFacts {
    let path = entry.storage_path();
    let bytes = Bytes::from(fs::read(dir.join(&path)).expect("read superfile"));
    let kvs = raw_footer_kvs(&bytes);
    let fts = first_region(&kvs, kv::FTS_OFFSET, kv::FTS_LENGTH)
        .unwrap_or_else(|| panic!("{path}: no FTS region"));
    let vec = first_region(&kvs, kv::VEC_OFFSET, kv::VEC_LENGTH);
    let columns_json = &kvs
        .iter()
        .find(|(k, _)| k == kv::FTS_COLUMNS)
        .unwrap_or_else(|| panic!("{path}: no {}", kv::FTS_COLUMNS))
        .1;
    let reader = SuperfileReader::open(bytes.clone()).expect("open superfile");
    let batch = reader.get_record_batch(None).expect("read rows");
    let ids = batch
        .column_by_name("_id")
        .expect("_id column")
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("_id is Decimal128")
        .values()
        .to_vec();
    SuperfileFacts {
        superfile_id: entry.superfile_id,
        bytes,
        fts,
        vec,
        ids,
        columns: serde_json::from_str(columns_json).expect("inf.fts.columns is JSON"),
        birth_version: entry.birth_version,
        partition_key: entry.partition_key.clone(),
    }
}

/// BM25 `k1` a reader assumes for a column entry written before the
/// parameters were recorded.
const UNRECORDED_K1: f64 = 1.2;
/// BM25 `b` a reader assumes for a column entry written before the
/// parameters were recorded.
const UNRECORDED_B: f64 = 0.75;

/// A column's `inf.fts.columns` entry without the two fields an analyzer
/// change is meant to move, and with the BM25 pair an older writer left
/// implicit spelled out, as today's writer records it.
fn without_analysis(column: &serde_json::Value) -> serde_json::Value {
    let mut column = column.clone();
    let fields = column.as_object_mut().expect("column entry is an object");
    fields.remove("tokenizer");
    fields.remove("analysis_revision");
    fields
        .entry("k1")
        .or_insert_with(|| serde_json::json!(UNRECORDED_K1));
    fields
        .entry("b")
        .or_insert_with(|| serde_json::json!(UNRECORDED_B));
    column
}

/// Every file under `dir`, recursively, relative to it and sorted; empty
/// when `dir` does not exist.
fn listing(dir: &Path) -> Vec<String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("read dir") {
            let path = entry.expect("dir entry").path();
            match path.is_dir() {
                true => walk(&path, out),
                false => out.push(path),
            }
        }
    }
    let mut files = Vec::new();
    if dir.is_dir() {
        walk(dir, &mut files);
    }
    let mut names: Vec<String> = files
        .iter()
        .map(|p| {
            p.strip_prefix(dir)
                .expect("under dir")
                .display()
                .to_string()
        })
        .collect();
    names.sort();
    names
}

/// The manifest list the table's pointer names, as raw JSON.
fn current_manifest_list(dir: &Path) -> serde_json::Value {
    let pointer = fs::read_to_string(dir.join("_supertable").join("current")).expect("pointer");
    let uri = pointer
        .lines()
        .find_map(|l| l.strip_prefix("manifest_uri="))
        .expect("pointer names a manifest");
    serde_json::from_slice(&fs::read(dir.join(uri)).expect("read manifest list"))
        .expect("manifest list is JSON")
}

/// `body` with every empty list dropped from its table records.
///
/// A record written by an older engine lacks the per-column lists added
/// since, and any catalog write by this one spells them out as empty; the
/// reader treats the two the same, so the comparison does too.
fn without_empty_lists(body: &serde_json::Value) -> serde_json::Value {
    let mut body = body.clone();
    if let Some(tables) = body["tables"].as_object_mut() {
        for record in tables.values_mut() {
            if let Some(fields) = record.as_object_mut() {
                fields.retain(|_, v| v.as_array().is_none_or(|list| !list.is_empty()));
            }
        }
    }
    body
}

/// The catalog body, as raw JSON.
fn catalog_json(root: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(root.join("_catalog").join("current")).expect("read catalog"))
        .expect("catalog is JSON")
}

/// After the change, every byte that should be carried is carried, every
/// byte that should change has changed, and nothing is left behind: the
/// superfiles' bodies, vectors and footers, the manifest, the deleted-rows
/// files, the catalog record, and the directories they live in, each read
/// back from disk rather than through the engine's own decoders.
fn assert_migration_leaves_no_debt(shape: &str) {
    let Some((_tmp, db, root)) = connect_corpus(shape) else {
        return;
    };
    let table = db.open_table(TABLE).expect("open");
    let dir = table_dir(&root);
    let hidden_dir = fs::read_dir(&dir)
        .expect("read table dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir() && p.to_string_lossy().ends_with("_vector_index"));

    let catalog_before = catalog_json(&root);
    let list_before = current_manifest_list(&dir);
    let hidden_before = hidden_dir.as_deref().map(listing).unwrap_or_default();
    let wal_before = listing(&dir.join("wal"));
    let inputs: HashMap<i128, SuperfileFacts> = table
        .local_handle()
        .reader()
        .expect("reader")
        .manifest()
        .get_all_superfiles()
        .iter()
        .map(|entry| {
            let facts = superfile_facts(&dir, entry);
            (facts.ids[0], facts)
        })
        .collect();

    table
        .reindex(&ReindexOptions::to_standard_analyzer())
        .expect("move to standard");
    table.gc(Duration::ZERO).expect("collect replaced files");

    let reader = table.local_handle().reader().expect("reader");
    let entries = reader.manifest().get_all_superfiles();
    assert_eq!(entries.len(), inputs.len(), "{shape}: one output per input");
    for entry in entries {
        let path = entry.storage_path();
        let out = superfile_facts(&dir, entry);
        let input = inputs
            .get(&out.ids[0])
            .unwrap_or_else(|| panic!("{path}: matches no input's rows"));
        assert_ne!(
            out.superfile_id, input.superfile_id,
            "{path}: not a new superfile"
        );
        assert_eq!(out.ids, input.ids, "{path}: rows or their order changed");
        assert_eq!(
            out.birth_version, input.birth_version,
            "{path}: birth version"
        );
        assert_eq!(out.partition_key, input.partition_key, "{path}: partition");

        // The Parquet body is carried byte for byte, so the FTS blob starts
        // where it did.
        let body_end = input.fts.0 as usize;
        assert_eq!(out.fts.0, input.fts.0, "{path}: body length changed");
        assert!(
            out.bytes[..body_end] == input.bytes[..body_end],
            "{path}: body bytes changed"
        );
        // The vector subsection is carried byte for byte, right after the
        // new FTS blob.
        match (input.vec, out.vec) {
            (None, None) => {}
            (Some((in_at, in_len)), Some((out_at, out_len))) => {
                assert_eq!(out_at, out.fts.0 + out.fts.1, "{path}: vector placement");
                assert!(
                    out.bytes[out_at as usize..(out_at + out_len) as usize]
                        == input.bytes[in_at as usize..(in_at + in_len) as usize],
                    "{path}: vector bytes changed"
                );
            }
            (before, after) => panic!("{path}: vector region {before:?} became {after:?}"),
        }

        // The FTS blob is the current format.
        let blob = &out.bytes[out.fts.0 as usize..];
        assert!(blob.starts_with(b"INFFTS01"), "{path}: no FTS magic");
        let version = u32::from_le_bytes(blob[8..12].try_into().expect("u32"));
        assert_eq!(version, VERSION_CURRENT, "{path}: FTS blob version");

        // Every footer key is stored once, and the builder is this engine.
        let kvs = raw_footer_kvs(&out.bytes);
        let mut keys: Vec<&str> = kvs.iter().map(|(k, _)| k.as_str()).collect();
        keys.sort_unstable();
        let unique = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), unique, "{path}: a footer key is stored twice");
        let builder = &kvs
            .iter()
            .find(|(k, _)| k == kv::BUILDER)
            .expect("builder key")
            .1;
        assert!(
            builder.contains(env!("CARGO_PKG_VERSION")),
            "{path}: written by {builder}"
        );

        // Each column: `standard`, its analysis recorded, everything else
        // as it was.
        assert_eq!(out.columns.len(), input.columns.len(), "{path}: columns");
        for (now, was) in out.columns.iter().zip(&input.columns) {
            assert_eq!(now["tokenizer"], "standard", "{path}: {now}");
            assert!(now["analysis_revision"].is_u64(), "{path}: {now}");
            assert_eq!(
                without_analysis(now),
                without_analysis(was),
                "{path}: a column setting other than the analyzer moved"
            );
        }

        // The manifest names the regions the footer does.
        let offsets = entry
            .subsection_offsets
            .as_ref()
            .unwrap_or_else(|| panic!("{path}: no subsection offsets"));
        assert_eq!(offsets.total_size, out.bytes.len() as u64, "{path}: size");
        assert_eq!(offsets.fts, Some(out.fts), "{path}: FTS region");
        assert_eq!(offsets.vec, out.vec, "{path}: vector region");
    }

    // On disk, the table's superfiles are exactly the live ones.
    let mut live: Vec<String> = entries.iter().map(|e| e.storage_path()).collect();
    live.sort();
    let mut on_disk: Vec<String> = superfile_paths(&dir)
        .iter()
        .map(|p| {
            p.strip_prefix(&dir)
                .expect("under table")
                .display()
                .to_string()
        })
        .collect();
    on_disk.sort();
    assert_eq!(
        on_disk, live,
        "{shape}: replaced or orphaned superfiles remain"
    );

    // The manifest list: new options, a complete term index that exists,
    // and deleted-rows files registered only for live superfiles — each of
    // which is on disk, with no file left for a replaced superfile.
    let list = current_manifest_list(&dir);
    assert_ne!(
        list["options_hash"], list_before["options_hash"],
        "{shape}: options hash"
    );
    assert_eq!(
        list["term_index_complete"], true,
        "{shape}: term index incomplete"
    );
    let index = list["term_index_uri"].as_str().expect("term index uri");
    assert!(
        dir.join(index).is_file(),
        "{shape}: term index {index} missing"
    );
    let live_ids: Vec<String> = entries.iter().map(|e| e.superfile_id.to_string()).collect();
    let mut registered: Vec<String> = list["tombstone_seqs"]
        .as_object()
        .expect("tombstone seqs")
        .keys()
        .cloned()
        .collect();
    registered.sort();
    let before_registered = list_before["tombstone_seqs"]
        .as_object()
        .map_or(0, |seqs| seqs.len());
    assert_eq!(
        registered.len(),
        before_registered,
        "{shape}: deleted-rows files were not carried one for one"
    );
    assert!(
        registered.iter().all(|id| live_ids.contains(id)),
        "{shape}: a deleted-rows file is registered for a replaced superfile"
    );
    let mut sidecars: Vec<String> = files_with_extension(&dir.join("superfiles"), "tombstones")
        .iter()
        .map(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .expect("sidecar name")
                .to_string()
        })
        .collect();
    sidecars.sort();
    assert_eq!(sidecars, registered, "{shape}: deleted-rows files on disk");

    // The catalog: `standard` recorded, nothing else about the table moved.
    let mut catalog = catalog_json(&root);
    let mut expected = catalog_before.clone();
    let analyzers = &mut expected["tables"][TABLE]["fts_analyzers"];
    for analyzer in analyzers.as_array_mut().expect("analyzer list") {
        *analyzer = serde_json::json!("standard");
    }
    catalog["catalog_id"] = serde_json::Value::Null;
    expected["catalog_id"] = serde_json::Value::Null;
    assert_eq!(
        without_empty_lists(&catalog),
        without_empty_lists(&expected),
        "{shape}: catalog record"
    );

    // Untouched: the hidden vector index and the mutation log.
    assert_eq!(
        hidden_dir.as_deref().map(listing).unwrap_or_default(),
        hidden_before,
        "{shape}: the hidden vector index changed"
    );
    assert_eq!(
        listing(&dir.join("wal")),
        wal_before,
        "{shape}: WAL state left"
    );

    let staleness = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess");
    assert!(
        staleness.is_current()
            && staleness.ascii_lower_columns.is_empty()
            && staleness.inconsistent_footers.is_empty(),
        "{shape}: {staleness:?}"
    );
    assert!(
        table
            .reindex_plan(&ReindexOptions::default())
            .expect("plan")
            .is_empty(),
        "{shape}: a default reindex still has work"
    );
}

#[test]
fn a_migrated_current_table_carries_no_debt() {
    assert_migration_leaves_no_debt("v7_ascii_lower");
}

#[test]
fn a_migrated_old_default_table_carries_no_debt() {
    assert_migration_leaves_no_debt("v4_bitset_blocks");
}
