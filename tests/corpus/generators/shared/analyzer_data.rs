// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

// Documents the `ascii_lower` and `standard` analyzers split differently,
// appended after the shared corpus, and the probes that tell the two apart.
//
// Shared verbatim between the generator that writes them and the tests that
// assert against them, for the same reason as `corpus_data.rs`: an expected
// count the tests keep their own copy of is one they can drift from.
//
// Every probe word is absent from the shared corpus, so its count is
// decided by these documents alone.

/// The `body` of each planted document, in append order. Row `i` of the
/// second append carries `ANALYZER_DOCS[i]`.
#[allow(dead_code)]
pub const ANALYZER_DOCS: &[&str] = &[
    // An apostrophe: `standard` keeps the word whole, `ascii_lower` splits
    // it at the apostrophe.
    "don't panic",
    // A decimal: `standard` keeps it whole, `ascii_lower` splits it at the
    // point.
    "version 3.14 released",
    // Accented letters: `ascii_lower` drops a token holding any non-ASCII
    // byte.
    "résumé attached",
    // An emoji: a term under `standard`, dropped by `ascii_lower`.
    "🚀 launch",
    // Removed by the generator after it is written, so the table carries a
    // tombstone. Holds a `standard`-only word: a re-analysis that dropped
    // the tombstone would bring it back.
    "über tombstoned",
];

/// Index into [`ANALYZER_DOCS`] of the document the generator deletes.
#[allow(dead_code)]
pub const DELETED_ANALYZER_DOC: usize = 4;

/// Title of planted document `i`, unique so the delete can name one row.
#[allow(dead_code)]
pub fn analyzer_title(i: usize) -> String {
    format!("planted p{i}")
}

/// A single-token probe of the `body` column, with the live documents it
/// matches under each analyzer.
///
/// Each word tokenizes to the same single term under both analyzers when
/// used as a query, so a count measures what the index holds rather than
/// how the query was split.
#[allow(dead_code)]
pub struct AnalyzerProbe {
    pub term: &'static str,
    pub ascii_lower_hits: usize,
    pub standard_hits: usize,
}

#[allow(dead_code)]
pub const ANALYZER_PROBES: &[AnalyzerProbe] = &[
    // The apostrophe's left half is a term only once `ascii_lower` splits.
    AnalyzerProbe {
        term: "don",
        ascii_lower_hits: 1,
        standard_hits: 0,
    },
    // The decimal's fractional half, likewise.
    AnalyzerProbe {
        term: "14",
        ascii_lower_hits: 1,
        standard_hits: 0,
    },
    AnalyzerProbe {
        term: "résumé",
        ascii_lower_hits: 0,
        standard_hits: 1,
    },
    AnalyzerProbe {
        term: "🚀",
        ascii_lower_hits: 0,
        standard_hits: 1,
    },
    // Deleted under both: the tombstone has to survive the migration.
    AnalyzerProbe {
        term: "über",
        ascii_lower_hits: 0,
        standard_hits: 0,
    },
    AnalyzerProbe {
        term: "tombstoned",
        ascii_lower_hits: 0,
        standard_hits: 0,
    },
    // Plain ASCII words: identical under both, so a migration must not
    // move them.
    AnalyzerProbe {
        term: "panic",
        ascii_lower_hits: 1,
        standard_hits: 1,
    },
    AnalyzerProbe {
        term: "launch",
        ascii_lower_hits: 1,
        standard_hits: 1,
    },
];
