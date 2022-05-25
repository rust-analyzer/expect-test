// Because this test modifies CARGO_MANIFEST_DIR, it has to be run as
// an integration test. This way we can isolate the modification to
// just one process.

use std::env;
use std::fs;
use std::path::Path;

use expect_test::ExpectFile;

#[test]
fn test_nested_workspaces() -> Result<(), std::io::Error> {
    let tmpdir = tempfile::tempdir()?;
    let outer_toml = tmpdir.path().join("Cargo.toml");
    let nested = tmpdir.path().join("nested");
    let nested_src = nested.join("src");
    let inner_toml = nested.join("Cargo.toml");

    fs::write(&outer_toml, r#"[package]\nname = "foo""#)?;

    fs::create_dir(&nested)?;
    fs::create_dir(&nested_src)?;
    fs::write(&inner_toml, r#"[package]\nname = "bar"\n[workspace]\n"#)?;

    // Fabricate a ExpectFile as if had been created with expect_file!
    // inside the nested project.
    env::set_var("CARGO_MANIFEST_DIR", nested.as_os_str());
    let expect_file = ExpectFile { path: Path::new("bar.txt").to_owned(), position: "src/lib.rs" };
    // Populate nested/src/bar.txt.
    env::set_var("UPDATE_EXPECT", "1");
    expect_file.assert_eq("Expected content");

    assert_eq!(fs::read_to_string(nested_src.join("bar.txt"))?, "Expected content");

    Ok(())
}
