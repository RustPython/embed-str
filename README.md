# embed-str: short string embedding for `Box<str>`


```rust
use embed_str::EmbeddingStr;

fn main() {
    let embedded = EmbeddingStr::from("short");
    let _s = embedded.as_str();  // &str

    let boxed = EmbeddingStr::from("long string is longer than limit");
    let _s = embedded.as_str();  // &str
}
```


- Documentation: [https://docs.rs/embed-str/](https://docs.rs/embed-str/)
- Crates: [https://crates.io/crates/embed-str](https://crates.io/crates/embed-str)