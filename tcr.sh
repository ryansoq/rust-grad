#!/bin/bash
# TCR: Test && Commit || Revert
# Usage: ./tcr.sh "commit message"
set -e
MSG="${1:-TCR: working}"
export PATH="$HOME/.cargo/bin:$PATH"
cargo test 2>&1 && {
    git add -A
    git -c user.name="Ryan" -c user.email="ryansoq@gmail.com" commit -m "$MSG"
    echo "✅ TCR: committed"
} || {
    git checkout -- .
    echo "❌ TCR: reverted"
    exit 1
}
