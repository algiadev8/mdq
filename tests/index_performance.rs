//! Reproducible, network-free indexing benchmark:
//! cargo test --release --test index_performance -- --ignored --nocapture
use std::time::Instant;

use mdq::db::Database;

#[test]
#[ignore = "manual indexing performance benchmark"]
fn benchmark_indexing() {
    let directory = tempfile::tempdir().unwrap();
    let vault = directory.path().join("vault");
    std::fs::create_dir(&vault).unwrap();
    let count = 2000;
    for i in 0..count {
        let mut text = format!("---\ntype: benchmark\nnumber: {i}\n---\n");
        for section in 0..6 {
            text.push_str(&format!("# Section {section}\n\n"));
            for _ in 0..4 {
                text.push_str("Rust indexing performance benchmark. 日本語の検索とインデックス計算を高速化する。\n\n");
            }
            text.push_str(&format!(
                "[[note-{}]] [next](note-{}.md)\n\n",
                (i + 1) % count,
                (i + 2) % count
            ));
        }
        std::fs::write(vault.join(format!("note-{i}.md")), text).unwrap();
    }
    let mut database = Database::open(&directory.path().join("index.sqlite3")).unwrap();
    for phase in ["initial", "unchanged", "one_changed"] {
        if phase == "one_changed" {
            std::fs::write(
                vault.join("note-0.md"),
                "# Updated\n\nChanged 日本語検索 [[note-1]]\n",
            )
            .unwrap();
        }
        let start = Instant::now();
        let stats = database.rebuild(&vault).unwrap();
        println!(
            "{phase}: {:.3} ms (notes={}, chunks={}, links={})",
            start.elapsed().as_secs_f64() * 1000.0,
            stats.notes,
            stats.chunks,
            stats.links
        );
        assert_eq!(stats.notes, count);
        assert_eq!(
            stats.chunks,
            if phase == "one_changed" { 11995 } else { 12000 }
        );
        assert_eq!(
            stats.links,
            if phase == "one_changed" { 3999 } else { 4000 }
        );
        assert!(!database.search("日本語", 10).unwrap().is_empty());
        assert_eq!(database.outgoing_links("note-1").unwrap().len(), 2);
    }
}
