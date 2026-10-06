// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Writes an `ascii_lower` corpus table with the 0.9.1 engine.
//!
//! The older `ascii_lower` shapes come from releases where it was the
//! default. This one is the table a current release writes when asked for
//! it explicitly. Its container is current and its analysis revision is
//! recorded, so the only thing a reindex could still change is the analyzer
//! itself. That is the migration this shape exists to exercise:
//!
//! - `body` is `ascii_lower`, and `title` is `ascii_lower` with positions,
//!   English stopwords and stemming, so a migration has to carry the
//!   filters across.
//! - `notes` is `standard` already, and a migration has to leave it alone.
//! - The shared corpus and the [`ANALYZER_DOCS`] go in as two appends, so
//!   the table holds more than one superfile and a migration has to publish
//!   them together.
//! - One planted document is deleted, so a migration has to carry a
//!   tombstone.
//!
//! Usage: `cargo run -- <output-dir> <table-name> [profile]`
//!
//! `profile` is `index_only` to write `body` without its text. Nothing can
//! re-analyze that column, so a migration has to refuse the table.

use std::{env, error::Error, sync::Arc};

use datafusion::prelude::{col, lit};
use infino::{
    FtsField, IndexSpec, Metric, Stemmer, Stopwords,
    arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch},
    arrow_schema::{DataType, Field, Schema, SchemaRef},
    connect,
};

include!("../../shared/corpus.rs");
include!("../../shared/analyzer_data.rs");

/// The `ascii_lower` base tokenizer's name.
const ASCII_LOWER: &str = "ascii_lower";
/// The only supported optional profile.
const PROFILE_INDEX_ONLY: &str = "index_only";
/// Command-line contract for this generator.
const USAGE: &str = "usage: <output-dir> <table-name> [index_only]";

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let out_dir = args.next().ok_or(USAGE)?;
    let table = args.next().ok_or(USAGE)?;
    let body_stored = match args.next().as_deref() {
        None => true,
        Some(PROFILE_INDEX_ONLY) => false,
        Some(other) => {
            return Err(
                format!("unknown profile {other:?}; expected {PROFILE_INDEX_ONLY:?}").into(),
            );
        }
    };
    if let Some(extra) = args.next() {
        return Err(format!("unexpected extra argument {extra:?}; {USAGE}").into());
    }

    let mut fields = text_fields();
    fields.push(embedding_field());
    let schema = Arc::new(Schema::new(fields));
    let spec = IndexSpec::new()
        .fts(
            FtsField::new("body")
                .analyzer(ASCII_LOWER)
                .stored(body_stored),
        )
        .fts(
            FtsField::new("title")
                .analyzer(ASCII_LOWER)
                .positions(true)
                .stopwords(Stopwords::English)
                .stemmer(Stemmer::English),
        )
        .fts(FtsField::new("notes"))
        .vector("emb", EMBEDDING_DIM, Metric::Cosine);

    let db = connect(&out_dir)?;
    let handle = db.create_table(&table, Arc::clone(&schema), spec)?;

    let mut shared: Vec<ArrayRef> = text_columns()
        .into_iter()
        .map(|c| Arc::new(c) as ArrayRef)
        .collect();
    shared.push(embeddings(0..N_DOCS)?);
    handle.append(&RecordBatch::try_new(Arc::clone(&schema), shared)?)?;

    handle.append(&planted_batch(&schema)?)?;
    // Titles are unique, so each delete has to land exactly one tombstone;
    // anything else means the table is not the one the tests expect.
    for doc in ANALYZER_DOCS.iter().filter(|doc| doc.deleted) {
        let deleted = handle.delete(col("title").eq(lit(doc.title)))?;
        if deleted.n_tombstoned() != 1 {
            return Err(format!(
                "tombstoned {} rows titled {:?}, expected 1",
                deleted.n_tombstoned(),
                doc.title
            )
            .into());
        }
    }

    println!(
        "wrote {} docs to {out_dir}/{table}, {} live",
        N_DOCS as usize + ANALYZER_DOCS.len(),
        N_DOCS as usize + LIVE_ANALYZER_DOCS
    );
    Ok(())
}

/// The [`ANALYZER_DOCS`], numbered on from the shared corpus.
fn planted_batch(schema: &SchemaRef) -> Result<RecordBatch, Box<dyn Error>> {
    let n = ANALYZER_DOCS.len();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(
            ANALYZER_DOCS.iter().map(|doc| doc.body).collect::<Vec<_>>(),
        )),
        Arc::new(LargeStringArray::from(
            ANALYZER_DOCS
                .iter()
                .map(|doc| doc.title)
                .collect::<Vec<_>>(),
        )),
        Arc::new(LargeStringArray::from(vec![None::<String>; n])),
        embeddings(N_DOCS..N_DOCS + n as u32)?,
    ];
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}
