// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! End-to-end supertable round-trip against a real GCS bucket.
//!
//! Gated on `INFINO_TEST_REAL_GCS=1` plus `INFINO_GCS_BUCKET` and
//! `GOOGLE_APPLICATION_CREDENTIALS` (a service-account key path). Every
//! storage call rides the real GCS HTTP wire; nothing short-circuits to the
//! local filesystem. Exercises the generation-keyed conditional-write path
//! (`cas_conformance`) and a full commit → reopen → query cycle through a
//! lazy disk cache, then deletes every object it wrote under its unique
//! prefix.
//!
//! No emulator variant: the common GCS emulators don't faithfully implement
//! the XML write API `object_store` uses (fake-gcs-server has no XML PUT;
//! storage-testbench's XML PUT omits the `ETag`/`x-goog-generation` response
//! headers `object_store` requires), so real GCS is the write-path gate.
//!
//! Invocation:
//!   INFINO_TEST_REAL_GCS=1 INFINO_GCS_BUCKET=<bucket> \
//!   GOOGLE_APPLICATION_CREDENTIALS=<sa-key.json> \
//!   cargo test -p infino --test supertable storage::real_gcs -- --nocapture

#![deny(clippy::unwrap_used)]

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use infino::{
    supertable::{
        Supertable,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        storage::{GcsStorageProvider, StorageProvider},
    },
    test_helpers::{
        build_title_batch, cas_conformance::cas_conformance, default_supertable_options,
    },
};
use tempfile::TempDir;

/// Disk-cache byte budget for the consumer (1 GiB; the fixture is tiny).
const CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Cold-fetch stream fan-out and chunk size for the consumer.
const COLD_FETCH_STREAMS: usize = 4;
const COLD_FETCH_CHUNK_BYTES: u64 = 1 << 20;

fn make_cache(
    storage: Arc<dyn StorageProvider>,
    cache_root: &std::path::Path,
) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("cache")
}

/// Real-GCS config from env: `(bucket, unique_prefix, sa_key_path)`. `None`
/// unless both `INFINO_GCS_BUCKET` and `GOOGLE_APPLICATION_CREDENTIALS` (a
/// service-account key path) are set. The prefix carries a per-run UUID so
/// concurrent/repeat runs never collide and cleanup stays scoped.
fn real_gcs_env() -> Option<(String, String, String)> {
    let bucket = std::env::var("INFINO_GCS_BUCKET").ok()?;
    let key_path = std::env::var("GOOGLE_APPLICATION_CREDENTIALS").ok()?;
    let root = std::env::var("INFINO_TEST_REAL_GCS_PREFIX")
        .unwrap_or_else(|_| "infino-real-gcs-integration".to_string());
    let prefix = format!("{}/{}", root.trim_matches('/'), uuid::Uuid::new_v4());
    Some((bucket, prefix, key_path))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_real_gcs_round_trip() {
    if std::env::var("INFINO_TEST_REAL_GCS").ok().as_deref() != Some("1") {
        eprintln!(
            "supertable_real_gcs_round_trip: skipped (set INFINO_TEST_REAL_GCS=1 + \
             INFINO_GCS_BUCKET + GOOGLE_APPLICATION_CREDENTIALS)"
        );
        return;
    }
    let Some((bucket, prefix, key_path)) = real_gcs_env() else {
        eprintln!(
            "supertable_real_gcs_round_trip: skipped (missing INFINO_GCS_BUCKET or \
             GOOGLE_APPLICATION_CREDENTIALS)"
        );
        return;
    };
    eprintln!("[real-gcs] bucket={bucket} prefix={prefix}");
    let opts = HashMap::from([("google_service_account".to_string(), key_path)]);

    // Prefix-scoped provider: every object this run writes lands under `prefix`.
    let storage: Arc<dyn StorageProvider> = Arc::new(
        GcsStorageProvider::new_with_prefix(&bucket, &prefix, &opts).expect("real gcs provider"),
    );

    // 1. Byte-level CAS conformance over the real GCS wire (generation-keyed;
    //    real GCS enforces if-generation-match, so stale rejection is asserted).
    cas_conformance(storage.as_ref(), "cas/conf", true).await;
    eprintln!("[real-gcs] CAS conformance OK");

    // 2. Commit → reopen → query through real GCS + a lazy disk cache.
    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create real gcs supertable");
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
            .expect("append");
        w.commit().expect("commit to real gcs");
        assert_eq!(producer.manifest_id(), 1);
    }
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = make_cache(Arc::clone(&storage), cache_dir.path());
    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("open real gcs supertable");
    assert_eq!(consumer.manifest_id(), 1);
    assert_eq!(consumer.reader().n_docs_total(), 2);
    let batches = consumer
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("query real gcs");
    assert_eq!(batches.len(), 1);
    assert!(
        cache.stats().n_cold_fetches >= 1,
        "reads must cold-fetch through GCS"
    );
    eprintln!(
        "[real-gcs] commit+reopen+query OK; n_cold_fetches={}",
        cache.stats().n_cold_fetches
    );

    // 3. Cleanup: a non-prefixed provider lists by absolute key and deletes
    //    every object under our unique prefix (list is absolute, delete on an
    //    empty-prefix provider is absolute — no double-prefixing).
    let cleanup: Arc<dyn StorageProvider> = Arc::new(
        GcsStorageProvider::new_with_prefix(&bucket, "", &opts).expect("cleanup provider"),
    );
    let keys = cleanup
        .list_with_prefix(&prefix)
        .await
        .expect("list cleanup");
    for key in &keys {
        cleanup.delete(key).await.expect("cleanup delete");
    }
    eprintln!(
        "[real-gcs] cleaned up {} objects under {prefix}",
        keys.len()
    );
}
