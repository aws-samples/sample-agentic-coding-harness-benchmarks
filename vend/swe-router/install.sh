#!/usr/bin/env bash
#
# Install the swe-router skill into a repository, or into a user-wide skills
# directory. The skill is five files with no dependencies, so this script only
# needs curl and a POSIX shell.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/aws-samples/sample-agentic-coding-harness-benchmarks/main/vend/swe-router/install.sh | bash
#   curl -sL .../install.sh | bash -s -- --dir ~/.claude/skills
#
# Every option also reads from an environment variable, so the piped form can
# be configured without -s -- when that is easier:
#   SWE_ROUTER_DIR=~/.claude/skills bash install.sh
#
# The download is not checksummed, because the files it fetches are the release:
# there is no separate artifact to check them against. Pass --ref <commit-sha> to
# install a fixed, reviewable revision rather than whatever main holds today.
#
set -euo pipefail

REPO="${SWE_ROUTER_REPO:-aws-samples/sample-agentic-coding-harness-benchmarks}"
REF="${SWE_ROUTER_REF:-main}"
SKILLS_DIR="${SWE_ROUTER_DIR:-.claude/skills}"
FORCE="${SWE_ROUTER_FORCE:-0}"

SKILL_NAME="swe-router"
SKILL_FILES="SKILL.md rubric.md route.py models.json model-aliases.json routing-policy.json allowed-models.txt"
JSON_FILES="models.json model-aliases.json routing-policy.json"
# The files a user is expected to edit. An upgrade preserves them, because
# overwriting either would silently replace an organisation's policy: which
# models it allows, and what quality each kind of task must reach.
POLICY_FILES="allowed-models.txt routing-policy.json"


usage() {
    cat <<'USAGE_EOF'
Install the swe-router skill.

Options:
  --dir DIR     Skills directory to install into (default: .claude/skills).
                Use ~/.claude/skills for a user-wide install.
  --ref REF     Git ref to install from (default: main). Pin a tag or commit
                to hold a version steady.
  --repo OWNER/NAME
                Source repository (default: aws-samples/sample-agentic-coding-harness-benchmarks).
  --force       Overwrite an existing allowed-models.txt and routing-policy.json.
                Without this, existing policy files are kept and the rest of
                the skill is still upgraded.
  -h, --help    Show this help.

Environment variables: SWE_ROUTER_DIR, SWE_ROUTER_REF, SWE_ROUTER_REPO,
SWE_ROUTER_FORCE. A command-line option wins over its variable.
USAGE_EOF
}


require_value() {
    # require_value <option-name> <count-of-remaining-args>
    if [ "$2" -lt 2 ]; then
        echo "install.sh: $1 needs a value" >&2
        exit 2
    fi
}


while [ $# -gt 0 ]; do
    case "$1" in
        --dir)   require_value "$1" "$#"; SKILLS_DIR="$2"; shift 2 ;;
        --ref)   require_value "$1" "$#"; REF="$2";        shift 2 ;;
        --repo)  require_value "$1" "$#"; REPO="$2";       shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "install.sh: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v curl >/dev/null 2>&1; then
    echo "install.sh: curl is required but was not found on PATH." >&2
    exit 1
fi

# The skill runs route.py, which needs only the Python standard library but
# does need python3, 3.8 or later, on PATH. Without it the skill cannot
# recommend anything, so refuse to install rather than leave a skill that
# fails the first time it fires. Nothing has been downloaded yet.
if ! command -v python3 >/dev/null 2>&1 \
        || ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    echo "install.sh: swe-router needs python3 (3.8 or later) on PATH, and none was found." >&2
    echo "It needs no Python packages, only the interpreter. Install one, then re-run:" >&2
    echo "  macOS:          xcode-select --install" >&2
    echo "  Debian/Ubuntu:  sudo apt install python3" >&2
    echo "  Windows:        use WSL, or install from python.org" >&2
    echo "Nothing was written." >&2
    exit 1
fi

BASE="https://raw.githubusercontent.com/${REPO}/${REF}/vend/${SKILL_NAME}"
TARGET="${SKILLS_DIR}/${SKILL_NAME}"

# Stage every file in a temp directory first, so a network failure part-way
# through leaves the existing install untouched rather than half-replaced.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "Installing ${SKILL_NAME} from ${REPO}@${REF} into ${TARGET}"

for f in $SKILL_FILES; do
    # --proto/--proto-redir pin the transfer to HTTPS: -L follows redirects, and
    # without these a redirect could walk the download down to plain HTTP, which
    # is a code download nobody would notice being tampered with.
    if ! curl -fsSL --proto '=https' --proto-redir '=https' \
            -o "${STAGE}/${f}" "${BASE}/${f}"; then
        echo "install.sh: could not download ${f} from ${BASE}/${f}" >&2
        echo "Nothing was written to ${TARGET}. Check --ref '${REF}' and your network." >&2
        exit 1
    fi
    if [ ! -s "${STAGE}/${f}" ]; then
        echo "install.sh: ${f} downloaded as an empty file; refusing to install." >&2
        exit 1
    fi
done

# A truncated JSON file makes the skill fail later with a confusing error, so
# catch it here while we can still say what went wrong.
if command -v python3 >/dev/null 2>&1; then
    for f in $JSON_FILES; do
        if ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "${STAGE}/${f}" 2>/dev/null; then
            echo "install.sh: ${f} is not valid JSON; the download was corrupted." >&2
            exit 1
        fi
    done
fi

mkdir -p "$TARGET"

kept=""
for f in $SKILL_FILES; do
    case " $POLICY_FILES " in
        *" $f "*)
            if [ -f "${TARGET}/${f}" ] && [ "$FORCE" != "1" ]; then
                kept="${kept} ${f}"
                continue
            fi
            ;;
    esac
    mv "${STAGE}/${f}" "${TARGET}/${f}"
done

chmod +x "${TARGET}/route.py"


echo
echo "Installed ${SKILL_NAME} to ${TARGET}"
if [ -n "$kept" ]; then
    echo "Kept your existing${kept}. Re-run with --force to replace them."
fi
case " $kept " in
    *" allowed-models.txt "*) ;;
    *)
        echo
        echo "Next step, and do not skip it: edit ${TARGET}/allowed-models.txt"
        echo "It ships allowing the five models on the measured frontier, one of them"
        echo "self-hosted. Left as-is in a setup that cannot reach that one, the skill"
        echo "will recommend a model nobody can select. Cut the list down to what your"
        echo "setup actually serves."
        ;;
esac
