pub fn logo() -> &'static str {
    r#"
     ___    __  __ ___ ___  __  __
    / _ \   \ \/ /|_ _/ _ \|  \/  |
   / /_\ \   >  <  | | | | | |\/| |
  /  _  \  /_/\_\| | |_| | |  | |
 /_/   \_\      |___\___/|_|  |_|
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logo_mentions_axiom_shape() {
        assert!(logo().contains("___"));
    }
}

