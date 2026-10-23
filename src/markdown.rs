use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use percent_encoding::percent_decode_str;
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use regex::Regex;
use sha2::{Digest, Sha256};

use crate::model::{ParsedChunk, ParsedLink, ParsedNote};

pub fn parse_note(vault: &Path, path: &Path) -> Result<ParsedNote> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let hash = hex::encode(Sha256::digest(&bytes));
    parse_note_bytes(vault, path, &bytes, hash)
}

/// Parse already-read bytes so incremental indexing does not read or hash them twice.
pub(crate) fn parse_note_bytes(
    vault: &Path,
    path: &Path,
    bytes: &[u8],
    hash: String,
) -> Result<ParsedNote> {
    let text = std::str::from_utf8(bytes)
        .with_context(|| format!("{} is not valid UTF-8", path.display()))?
        .replace("\r\n", "\n");
    let metadata = fs::metadata(path)?;
    let mtime = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();
    let ctime = metadata
        .created()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(mtime);
    let relative = path
        .strip_prefix(vault)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    let title = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_owned();
    let (frontmatter, body, body_start_line) = split_frontmatter(&text);
    let link_source = frontmatter
        .as_ref()
        .map(|value| format!("{body}\n{}", value))
        .unwrap_or_else(|| body.to_owned());

    Ok(ParsedNote {
        path: relative,
        title,
        frontmatter,
        chunks: split_chunks(body),
        links: extract_links(&link_source),
        body: body.to_owned(),
        body_start_line,
        mtime,
        ctime,
        size: metadata.len(),
        hash,
    })
}

fn split_frontmatter(text: &str) -> (Option<serde_json::Value>, &str, usize) {
    let Some(rest) = text.strip_prefix("---\n") else {
        return (None, text, 1);
    };
    let Some(end) = rest.find("\n---\n") else {
        return (None, text, 1);
    };
    let yaml = &rest[..end];
    let body = &rest[end + 5..];
    let body_start_index = text.len() - body.len();
    let body_start_line = text[..body_start_index]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1;
    let value = serde_yaml::from_str::<serde_yaml::Value>(yaml)
        .ok()
        .and_then(|value| serde_json::to_value(value).ok());
    (value, body, body_start_line)
}

fn split_chunks(body: &str) -> Vec<ParsedChunk> {
    static HEADING_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$").unwrap());
    let heading_re = &*HEADING_RE;
    let mut sections = Vec::new();
    let mut last = 0;
    let mut heading = None;

    for captures in heading_re.captures_iter(body) {
        let matched = captures.get(0).unwrap();
        push_section(&mut sections, heading.take(), &body[last..matched.start()]);
        heading = Some(captures[2].trim().to_owned());
        last = matched.end();
    }
    push_section(&mut sections, heading, &body[last..]);

    if sections.is_empty() && !body.trim().is_empty() {
        sections.push(ParsedChunk {
            ordinal: 0,
            heading: None,
            body: body.trim().to_owned(),
        });
    }

    for (ordinal, chunk) in sections.iter_mut().enumerate() {
        chunk.ordinal = ordinal;
    }
    sections
}

fn push_section(chunks: &mut Vec<ParsedChunk>, heading: Option<String>, text: &str) {
    const MAX_CHARS: usize = 2400;
    let text = text.trim();
    if text.is_empty() {
        return;
    }

    let mut current = String::new();
    let mut current_chars = 0;
    for paragraph in text.split("\n\n") {
        let paragraph_chars = paragraph.chars().count();
        if !current.is_empty() && current_chars + paragraph_chars > MAX_CHARS {
            chunks.push(ParsedChunk {
                ordinal: 0,
                heading: heading.clone(),
                body: current.trim().to_owned(),
            });
            current.clear();
            current_chars = 0;
        }
        if !current.is_empty() {
            current.push_str("\n\n");
            current_chars += 2;
        }
        current.push_str(paragraph);
        current_chars += paragraph_chars;
    }
    if !current.trim().is_empty() {
        chunks.push(ParsedChunk {
            ordinal: 0,
            heading,
            body: current.trim().to_owned(),
        });
    }
}

pub fn extract_links(body: &str) -> Vec<ParsedLink> {
    // pulldown-cmark deliberately exposes code as text-like events. Mask those byte
    // ranges before applying the Wiki-link regexp so links shown as examples never
    // become graph edges. Newlines are retained to keep the Markdown structure stable.
    let mut masked = body.contains("[[").then(|| body.as_bytes().to_vec());
    let mut markdown_links = Vec::new();
    let mut in_code_block = false;
    for (event, range) in Parser::new_ext(body, Options::all()).into_offset_iter() {
        match event {
            Event::Start(Tag::CodeBlock(_)) => in_code_block = true,
            Event::End(TagEnd::CodeBlock) => {
                if let Some(masked) = &mut masked {
                    mask_bytes(masked, range);
                }
                in_code_block = false;
                continue;
            }
            Event::Code(_) | Event::InlineMath(_) | Event::DisplayMath(_) => {
                if let Some(masked) = &mut masked {
                    mask_bytes(masked, range);
                }
                continue;
            }
            Event::Start(Tag::Link { dest_url, .. }) => {
                if let Some(link) = markdown_link(&dest_url, false) {
                    markdown_links.push(link);
                }
            }
            Event::Start(Tag::Image { dest_url, .. }) => {
                if let Some(link) = markdown_link(&dest_url, true) {
                    markdown_links.push(link);
                }
            }
            _ => {}
        }
        if in_code_block {
            if let Some(masked) = &mut masked {
                mask_bytes(masked, range);
            }
        }
    }
    let wiki_source = masked
        .map(|masked| String::from_utf8(masked).expect("masking produces valid UTF-8"))
        .unwrap_or_default();
    static WIKI_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(!)?\[\[([^\[\]]+)\]\]").unwrap());
    let wiki_re = &*WIKI_RE;
    let mut links: Vec<ParsedLink> = wiki_re
        .captures_iter(&wiki_source)
        .filter_map(|captures| {
            let raw = captures.get(2)?.as_str().trim().to_owned();
            let destination = raw.split('|').next()?.trim().to_owned();
            let (target, heading) = destination
                .split_once('#')
                .map(|(target, heading)| (target.trim(), Some(heading.trim().to_owned())))
                .unwrap_or((destination.as_str(), None));
            if target.is_empty() {
                return None;
            }
            Some(ParsedLink {
                raw_target: raw,
                target: normalize_target(target),
                heading,
                is_embed: captures.get(1).is_some(),
            })
        })
        .collect();

    // Preserve the original Wiki-first order, including which duplicate wins.
    links.extend(markdown_links);
    let mut seen = HashSet::new();
    links.retain(|link| seen.insert((link.target.clone(), link.heading.clone(), link.is_embed)));
    links
}

fn markdown_link(destination: &str, is_embed: bool) -> Option<ParsedLink> {
    let raw = destination.trim().to_owned();
    if is_external_destination(&raw) {
        return None;
    }
    let decoded = percent_decode_str(&raw).decode_utf8_lossy().into_owned();
    let (target, heading) = decoded
        .split_once('#')
        .map(|(target, heading)| (target.trim(), Some(heading.trim().to_owned())))
        .unwrap_or((decoded.as_str(), None));
    if target.is_empty() {
        return None;
    }
    Some(ParsedLink {
        raw_target: raw,
        target: normalize_target(target),
        heading,
        is_embed,
    })
}

fn mask_bytes(bytes: &mut [u8], range: std::ops::Range<usize>) {
    for byte in &mut bytes[range] {
        if *byte != b'\n' && *byte != b'\r' {
            *byte = b' ';
        }
    }
}

pub fn extract_tags(body: &str) -> Vec<String> {
    static TAG_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?:^|[^A-Za-z0-9_])#([A-Za-z0-9_/-]+)").unwrap());
    let tag_re = &*TAG_RE;
    let mut tags: Vec<String> = tag_re
        .captures_iter(body)
        .filter_map(|captures| {
            let tag = captures.get(1)?.as_str().trim_matches('/').to_owned();
            if tag.is_empty() { None } else { Some(tag) }
        })
        .collect();
    let mut seen = HashSet::new();
    tags.retain(|tag| seen.insert(tag.clone()));
    tags
}

pub fn normalize_target(target: &str) -> String {
    let mut path = PathBuf::from(target.trim());
    if path
        .extension()
        .is_some_and(|extension| extension.to_string_lossy().eq_ignore_ascii_case("md"))
    {
        path.set_extension("");
    }
    path.to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("./")
        .trim_matches('/')
        .to_lowercase()
}

fn is_external_destination(destination: &str) -> bool {
    let lowercase = destination.to_ascii_lowercase();
    lowercase.starts_with("//") || has_uri_scheme(&lowercase)
}

fn has_uri_scheme(destination: &str) -> bool {
    let Some((scheme, _)) = destination.split_once(':') else {
        return false;
    };
    let mut chars = scheme.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphabetic()
        && chars.clone().count() > 0
        && chars.all(|char| char.is_ascii_alphanumeric() || matches!(char, '+' | '-' | '.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_ignore_inline_and_fenced_code() {
        let body = "[[kept|Alias]] `[[inline]]`\n~~~md\n[[fenced]]\n```\n[[nested-looking]]\n```\n~~~\n[[after#Heading]]";
        let links = extract_links(body);
        assert_eq!(
            links
                .iter()
                .map(|link| link.target.as_str())
                .collect::<Vec<_>>(),
            ["kept", "after"]
        );
        assert_eq!(links[1].heading.as_deref(), Some("Heading"));
    }

    #[test]
    fn links_ignore_unclosed_fence_and_markdown_links_in_code() {
        let body = "[kept](target.md)\n```\n[[wiki-code]]\n[code](other.md)";
        let links = extract_links(body);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "target");
    }

    #[test]
    fn fused_link_scan_preserves_wiki_first_deduplication_and_math_masking() {
        let links = extract_links(
            "[first](target.md#Part) [[target#Part|Alias]] $[[math]]$ ![asset](a.png) ![[a.png]] [[最後]]",
        );
        assert_eq!(
            links
                .iter()
                .map(|link| link.raw_target.as_str())
                .collect::<Vec<_>>(),
            ["target#Part|Alias", "a.png", "最後"]
        );
        assert_eq!(links[0].heading.as_deref(), Some("Part"));
        assert!(links[1].is_embed);
        let markdown_only = extract_links("[kept](target.md) `code` ![image](a.png)");
        assert_eq!(markdown_only.len(), 2);
        assert_eq!(markdown_only[0].target, "target");
    }

    #[test]
    fn chunk_character_counter_preserves_threshold_and_unicode_boundaries() {
        let first = "日".repeat(1200);
        let second = "本".repeat(1200);
        // As before, the separator is not counted until after the threshold check.
        let body = format!("{first}\n\n{second}\n\nx");
        let chunks = split_chunks(&body);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].body, format!("{first}\n\n{second}"));
        assert_eq!(chunks[1].body, "x");
        assert_eq!(chunks[1].ordinal, 1);
    }

    #[test]
    fn parses_arbitrary_frontmatter() {
        let text = "---\ncustom:\n  nested: 42\n旗: true\n---\n# Body\nText";
        let (frontmatter, body, body_start_line) = split_frontmatter(text);
        let value = frontmatter.unwrap();
        assert_eq!(value["custom"]["nested"], 42);
        assert_eq!(value["旗"], true);
        assert!(body.contains("# Body"));
        assert_eq!(body_start_line, 6);
    }

    #[test]
    fn extracts_wiki_links_without_property_assumptions() {
        let links = extract_links("See [[Folder/Note#Section|label]] and ![[asset.png]].");
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].target, "folder/note");
        assert_eq!(links[0].heading.as_deref(), Some("Section"));
        assert!(links[1].is_embed);
    }

    #[test]
    fn extracts_markdown_links_and_decodes_paths() {
        let links = extract_links(
            "[note](<../Folder/A Note.md#Part> \"title\") ![embed](asset.png) [web](https://example.com) [zotero](zotero://select/library/items/ABC123)",
        );
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].target, "../folder/a note");
        assert_eq!(links[0].heading.as_deref(), Some("Part"));
        assert!(links[1].is_embed);
    }

    #[test]
    fn ignores_markdown_links_with_arbitrary_uri_schemes() {
        let links = extract_links(
            "[zotero](zotero://select/library/items/ABC123) [obsidian](obsidian://open?vault=Notes) [tel](tel:+123456789) [path](folder:note.md) [windows](C:\\Users\\note.md)",
        );
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "c:/users/note");
    }

    #[test]
    fn extracts_inline_tags_without_headings() {
        let tags = extract_tags("# Heading\nText #project/alpha and #work.");
        assert_eq!(tags, vec!["project/alpha", "work"]);
    }

    #[test]
    fn rejects_non_utf8_markdown_instead_of_lossy_indexing() {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("vault");
        fs::create_dir_all(&vault).unwrap();
        let note = vault.join("latin1.md");
        fs::write(&note, b"# Latin1\n- [ ] Caf\xe9\n").unwrap();

        let error = parse_note(&vault, &note).unwrap_err();

        assert!(error.to_string().contains("is not valid UTF-8"));
    }
}
