pub mod dom;
pub mod html;

#[cfg(test)]
mod tests {
    use crate::html::parse;

    const EXAMPLE_COM_HTML: &str = r#"<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style type="text/css">
    body {
        background-color: #f0f0f2;
        margin: 0;
        padding: 0;
        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    </style>
</head>
<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents. You may use this
    domain in literature without prior coordination or asking for permission.</p>
    <p><a href="https://www.iana.org/domains/example">More information...</a></p>
</div>
</body>
</html>
"#;

    #[test]
    fn parse_example_com_and_extract_title() {
        let doc = parse(EXAMPLE_COM_HTML);
        let title_nodes = doc.query_selector("title");
        assert_eq!(title_nodes.len(), 1, "expected exactly one <title> element");

        let title_el = &title_nodes[0];
        let text = title_el.text_content();
        assert_eq!(text, "Example Domain");
    }

    #[test]
    fn query_selector_finds_multiple_paragraphs() {
        let doc = parse(EXAMPLE_COM_HTML);
        let ps = doc.query_selector("p");
        assert_eq!(ps.len(), 2);
    }

    #[test]
    fn malformed_unclosed_tag_recovery() {
        let html = r#"<html><body><p>hello <b>world</body></html>"#;
        let doc = parse(html);
        let bs = doc.query_selector("b");
        assert_eq!(bs.len(), 1);
        assert_eq!(bs[0].text_content(), "world");
    }

    #[test]
    fn malformed_missing_closing_html() {
        let html = r#"<html><head><title>T</title></head><body></body>"#;
        let doc = parse(html);
        let ts = doc.query_selector("title");
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0].text_content(), "T");
    }

    #[test]
    fn comment_is_ignored_in_tree() {
        let html = r#"<html><!-- secret --><body><p>visible</p></body></html>"#;
        let doc = parse(html);
        let ps = doc.query_selector("p");
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text_content(), "visible");
    }

    #[test]
    fn self_closing_tag_handled() {
        let html = r#"<html><body><img src="x.png" /><br><p>hi</p></body></html>"#;
        let doc = parse(html);
        let ps = doc.query_selector("p");
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text_content(), "hi");
    }
}
