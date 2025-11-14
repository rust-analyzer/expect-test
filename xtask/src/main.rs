use std::{env, fs, path::PathBuf};

use xshell::{cmd, Shell};

type Error = Box<dyn std::error::Error>;
type Result<T, E = Error> = std::result::Result<T, E>;

fn main() {
    if let Err(err) = try_main() {
        eprintln!("error: {}", err);
        std::process::exit(1)
    }
}

fn try_main() -> Result<()> {
    let subcommand = std::env::args().nth(1);
    match subcommand {
        Some(it) if it == "ci" => (),
        _ => {
            print_usage();
            Err("invalid arguments")?
        }
    }

    let sh = Shell::new()?;

    let cargo_toml = cargo_toml(&sh)?;
    let version = cargo_toml.get("version")?;
    let tag = format!("v{}", version);

    let dry_run =
        env::var("CI").is_err() || git::has_tag(&sh, &tag)? || git::current_branch(&sh)? != "master";

    let token = env::var("CRATES_IO_TOKEN").unwrap_or("no token".to_string());
    let dry_run_flag = dry_run.then_some("--dry-run");
    cmd!(sh, "cargo publish --token {token} {dry_run_flag...}").run()?;

    if !dry_run {
        cmd!(sh, "git tag {tag}").run()?;
        cmd!(sh, "git push --tags").run()?;
    }

    Ok(())
}

pub fn cargo_toml(sh: &Shell) -> Result<CargoToml> {
    let cwd = sh.current_dir();
    let path = cwd.join("Cargo.toml");
    let contents = fs::read_to_string(&path)?;
    Ok(CargoToml { path, contents })
}

pub struct CargoToml {
    path: PathBuf,
    contents: String,
}

impl CargoToml {
    fn get(&self, field: &str) -> Result<&str> {
        for line in self.contents.lines() {
            let words = line.split_ascii_whitespace().collect::<Vec<_>>();
            match words.as_slice() {
                [n, "=", v, ..] if n.trim() == field => {
                    assert!(v.starts_with('"') && v.ends_with('"'));
                    return Ok(&v[1..v.len() - 1]);
                }
                _ => (),
            }
        }
        Err(format!("can't find `{}` in {}", field, self.path.display()))?
    }
}

fn print_usage() {
    eprintln!(
        "\
Usage: cargo run -p xtask <SUBCOMMAND>

SUBCOMMANDS:
    ci
"
    )
}

mod git {
    use crate::Result;
    use xshell::{cmd, Shell};

    pub(crate) fn current_branch(sh: &Shell) -> Result<String> {
        let res = cmd!(sh, "git branch --show-current").read()?;
        Ok(res)
    }

    pub(crate) fn has_tag(sh: &Shell, tag: &str) -> Result<bool> {
        let res = tag_list(sh)?.iter().any(|it| it == tag);
        Ok(res)
    }

    fn tag_list(sh: &Shell) -> Result<Vec<String>> {
        let tags = cmd!(sh, "git tag --list").read()?;
        let res = tags.lines().map(|it| it.trim().to_string()).collect();
        Ok(res)
    }
}
