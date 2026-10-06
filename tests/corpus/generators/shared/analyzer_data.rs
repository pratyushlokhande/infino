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

/// One planted row appended after the shared corpus.
#[allow(dead_code)]
pub struct AnalyzerDoc {
    pub title: &'static str,
    pub body: &'static str,
    pub deleted: bool,
}

/// The planted documents, in append order.
#[allow(dead_code)]
pub const ANALYZER_DOCS: &[AnalyzerDoc] = &[
    AnalyzerDoc {
        title: "planted p0",
        // An apostrophe: `standard` keeps the word whole, `ascii_lower`
        // splits it at the apostrophe.
        body: "don't panic",
        deleted: false,
    },
    AnalyzerDoc {
        title: "planted p1",
        // A decimal: `standard` keeps it whole, `ascii_lower` splits it at
        // the point.
        body: "version 3.14 released",
        deleted: false,
    },
    AnalyzerDoc {
        title: "planted p2",
        // Accented letters: `ascii_lower` drops a token holding any
        // non-ASCII byte.
        body: "résumé attached",
        deleted: false,
    },
    AnalyzerDoc {
        title: "planted p3",
        // An emoji: a term under `standard`, dropped by `ascii_lower`.
        body: "🚀 launch",
        deleted: false,
    },
    AnalyzerDoc {
        title: "planted p4",
        // Removed by the generator after it is written, so the table carries
        // a tombstone. Holds a `standard`-only word: a re-analysis that
        // dropped the tombstone would bring it back.
        body: "über tombstoned",
        deleted: true,
    },
];

/// Planted rows left live: every one not marked `deleted`.
#[allow(dead_code)]
pub const LIVE_ANALYZER_DOCS: usize = {
    let mut live = 0;
    let mut i = 0;
    while i < ANALYZER_DOCS.len() {
        if !ANALYZER_DOCS[i].deleted {
            live += 1;
        }
        i += 1;
    }
    live
};

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
