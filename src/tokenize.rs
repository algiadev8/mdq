pub fn search_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut cjk = String::new();

    let flush_word = |word: &mut String, tokens: &mut Vec<String>| {
        if !word.is_empty() {
            tokens.push(std::mem::take(word));
        }
    };
    let flush_cjk = |cjk: &mut String, tokens: &mut Vec<String>| {
        if cjk.is_empty() {
            return;
        }
        let chars: Vec<char> = cjk.chars().collect();
        if chars.len() == 1 {
            tokens.push(chars[0].to_string());
        } else {
            for pair in chars.windows(2) {
                tokens.push(pair.iter().collect());
            }
        }
        cjk.clear();
    };

    for character in text.chars() {
        if is_cjk(character) {
            flush_word(&mut word, &mut tokens);
            cjk.push(character);
        } else if character.is_alphanumeric() || character == '_' {
            flush_cjk(&mut cjk, &mut tokens);
            for lowercase in character.to_lowercase() {
                word.push(lowercase);
            }
        } else {
            flush_word(&mut word, &mut tokens);
            flush_cjk(&mut cjk, &mut tokens);
        }
    }
    flush_word(&mut word, &mut tokens);
    flush_cjk(&mut cjk, &mut tokens);
    tokens
}

pub fn index_text(text: &str) -> String {
    // Write directly into one buffer instead of allocating a String per token.
    // Keep duplicates and the exact same token boundaries as search_tokens().
    let mut output = String::with_capacity(text.len());
    let mut in_word = false;
    let mut previous_cjk = None;
    let mut paired = false;
    for character in text.chars() {
        if is_cjk(character) {
            in_word = false;
            if let Some(previous) = previous_cjk {
                token_separator(&mut output);
                output.push(previous);
                output.push(character);
                paired = true;
            }
            previous_cjk = Some(character);
        } else {
            if let Some(previous) = previous_cjk.take() {
                if !paired {
                    token_separator(&mut output);
                    output.push(previous);
                }
                paired = false;
            }
            if character.is_alphanumeric() || character == '_' {
                if !in_word {
                    token_separator(&mut output);
                    in_word = true;
                }
                output.extend(character.to_lowercase());
            } else {
                in_word = false;
            }
        }
    }
    if let Some(previous) = previous_cjk {
        if !paired {
            token_separator(&mut output);
            output.push(previous);
        }
    }
    output
}

fn token_separator(output: &mut String) {
    if !output.is_empty() {
        output.push(' ');
    }
}

pub fn fts_query(text: &str) -> String {
    search_tokens(text)
        .into_iter()
        .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn is_cjk(character: char) -> bool {
    matches!(
        character as u32,
        0x3040..=0x30ff
            | 0x3400..=0x4dbf
            | 0x4e00..=0x9fff
            | 0xf900..=0xfaff
            | 0xac00..=0xd7af
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_index_matches_reference_tokenizer() {
        let alphabet = [
            'A', 'z', '_', '7', ' ', '\n', '"', '日', '本', 'あ', 'ア', '한', 'é', 'İ', '\u{301}',
            '😀',
        ];
        // Exhaust every four-character combination, including Unicode lowercase
        // expansion, CJK/Latin transitions, punctuation and isolated CJK characters.
        for &a in &alphabet {
            for &b in &alphabet {
                for &c in &alphabet {
                    for &d in &alphabet {
                        let text: String = [a, b, c, d].into_iter().collect();
                        assert_eq!(
                            index_text(&text),
                            search_tokens(&text).join(" "),
                            "{text:?}"
                        );
                    }
                }
            }
        }
        assert_eq!(index_text(""), "");
    }

    #[test]
    fn tokenizes_japanese_as_bigrams() {
        assert_eq!(search_tokens("格子暗号"), vec!["格子", "子暗", "暗号"]);
    }

    #[test]
    fn tokenizes_latin_words_case_insensitively() {
        assert_eq!(search_tokens("Rust CLI"), vec!["rust", "cli"]);
    }

    #[test]
    fn fts_query_joins_tokens_with_or() {
        assert_eq!(fts_query("benchmark design"), "\"benchmark\" OR \"design\"");
    }

    #[test]
    fn fts_query_single_token() {
        assert_eq!(fts_query("benchmark"), "\"benchmark\"");
    }

    #[test]
    fn fts_query_escapes_quotes() {
        assert_eq!(fts_query("hello\"world"), "\"hello\" OR \"world\"");
    }
}
