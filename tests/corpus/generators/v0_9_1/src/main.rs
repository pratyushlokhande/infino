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

use std::{env, sync::Arc};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let out_dir = args
        .next()
        .ok_or("usage: <output-dir> <table-name> [profile]")?;
    let table = args
        .next()
        .ok_or("usage: <output-dir> <table-name> [profile]")?;
    let body_stored = args.next().as_deref() != Some("index_only");

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
    let deleted = handle.delete(col("title").eq(lit(analyzer_title(DELETED_ANALYZER_DOC))))?;
    if deleted.n_tombstoned() != 1 {
        return Err(format!(
            "tombstoned {} planted rows, expected 1",
            deleted.n_tombstoned()
        )
        .into());
    }

    println!(
        "wrote {} docs to {out_dir}/{table}",
        N_DOCS as usize + ANALYZER_DOCS.len()
    );
    Ok(())
}

fn embedding_field() -> Field {
    Field::new(
        "emb",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            EMBEDDING_DIM as i32,
        ),
        false,
    )
}

/// The `emb` column for documents `ids`.
fn embeddings(ids: impl Iterator<Item = u32>) -> Result<ArrayRef, Box<dyn std::error::Error>> {
    let flat: Vec<f32> = ids.flat_map(embedding).collect();
    Ok(Arc::new(FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        EMBEDDING_DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )?))
}

/// The [`ANALYZER_DOCS`], numbered on from the shared corpus.
fn planted_batch(schema: &SchemaRef) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let n = ANALYZER_DOCS.len();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(ANALYZER_DOCS.to_vec())),
        Arc::new(LargeStringArray::from(
            (0..n).map(analyzer_title).collect::<Vec<_>>(),
        )),
        Arc::new(LargeStringArray::from(vec![None::<String>; n])),
        embeddings(N_DOCS..N_DOCS + n as u32)?,
    ];
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}
