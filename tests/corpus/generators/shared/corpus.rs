// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

// The writing half of the shared corpus: the schema, batch and append
// every generator repeats. Includes the documents themselves.

include!("corpus_data.rs");

/// The three text columns every generated table carries.
///
/// Held here rather than restated per generator: the corpus is only a
/// controlled comparison if the tables differ in format shape and nothing
/// else, and a schema copied six times is six chances to break that.
#[allow(dead_code)]
pub fn text_fields() -> Vec<Field> {
    vec![
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("notes", DataType::LargeUtf8, true),
    ]
}

/// The text columns' data, in [`text_fields`] order.
#[allow(dead_code)]
pub fn text_columns() -> Vec<LargeStringArray> {
    vec![
        LargeStringArray::from((0..N_DOCS).map(body).collect::<Vec<_>>()),
        LargeStringArray::from((0..N_DOCS).map(title).collect::<Vec<_>>()),
        LargeStringArray::from((0..N_DOCS).map(notes).collect::<Vec<_>>()),
    ]
}

/// The shared embedding column every vector-bearing corpus generator uses.
#[allow(dead_code)]
pub fn embedding_field() -> Field {
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
#[allow(dead_code)]
pub fn embeddings(
    ids: impl Iterator<Item = u32>,
) -> Result<ArrayRef, Box<dyn std::error::Error>> {
    let flat: Vec<f32> = ids.flat_map(embedding).collect();
    Ok(Arc::new(FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        EMBEDDING_DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )?))
}

/// Write the shared corpus as a text-only table indexed by `spec`.
///
/// `spec` stays the caller's because it is the one thing that genuinely
/// differs: the `fts` setter takes a `&str` on the older engines and an
/// `FtsField` on the newer ones, and only some expose positions at all.
#[allow(dead_code)]
pub fn write_text_corpus(
    out_dir: &str,
    table: &str,
    spec: IndexSpec,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(text_fields()));
    let db = connect(out_dir)?;
    let handle = db.create_table(table, Arc::clone(&schema), spec)?;
    let columns: Vec<_> = text_columns()
        .into_iter()
        .map(|c| Arc::new(c) as _)
        .collect();
    handle.append(&RecordBatch::try_new(schema, columns)?)?;
    println!("wrote {N_DOCS} docs to {out_dir}/{table}");
    Ok(())
}
