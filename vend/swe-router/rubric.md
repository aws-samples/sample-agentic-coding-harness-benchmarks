# Task rubric

Score the task on six dimensions, each 0, 1 or 2. The scores decide which model gets the work, so each one must describe the task as it would land in **this** repository, not in the abstract.

## Before you score

Read the repository's own map first: `AGENTS.md` at the root, else `CLAUDE.md`, else `README.md`. Follow the documents it links to. It tells you the stack, the layout, where the tests live and what the project treats as risky. If the repository has a `SECURITY.md`, read it too: it says what the project treats as security-sensitive, which bears on silent invariant and blast radius. Where any of these disagree with the code, trust the code.

Then read the task, and look at the parts of the repository it touches: the files it names or implies, the tests around them, and anything those files depend on or are depended on by. You are judging the task, so do not write or change code.

Three rules apply to every dimension:

- **Size is not a signal.** Line count, number of files, length of the task text and how long the work might take do not raise or lower any score. A one-line change can score 2; a twenty-file change can score 0.
- **Score what the task needs, not what a model might get wrong.** You are describing the work, not predicting a failure.
- **Every score needs evidence.** Name the file, document, test, issue or sentence of the task that justifies it. A score you cannot point to is a guess; say so and lower your confidence.

## 1. Hidden contract

**Question:** Is the correct behaviour fully stated in the task, or does it depend on something defined elsewhere?

**Look at:** what the task links to or mentions (an earlier issue, a design doc, an RFC, another service's API); comments and docs near the code it touches; tests that pin current behaviour other parts rely on; how the repository map describes the interfaces between components.

- **0:** The task states the behaviour completely. You could write the acceptance test from the task text alone.
- **1:** The behaviour follows from nearby code or project conventions the task does not mention, such as a naming pattern, an existing error format, or how sibling commands already behave. Reading the surrounding code reveals it.
- **2:** The behaviour is fixed by something outside this change: an earlier issue or design decision, another service's contract, a spec, or callers that depend on it. A change that ignores it can look complete and still contradict it.

**Common mistake:** scoring 0 because the task is short. "Implement the gate agreed in #809" is a 2.

## 2. Silent invariant

**Question:** Could a wrong change look like it works?

**Look at:** whether the code touches authentication, authorisation, input validation, secrets, money, persisted data, migrations, caching, concurrency or retries; what the repository map calls security-critical; whether a mistake would show up in normal use or only under a particular input, timing or attacker.

- **0:** A wrong change is visible: an error, a wrong value on screen, a failing command.
- **1:** A wrong change shows only in edge cases a normal check would miss: an unusual input, an alternate configuration, a second platform.
- **2:** Correctness hinges on a security, data-integrity or concurrency invariant, and breaking it is silent: the feature appears to work while it leaks, corrupts, races or lets the wrong caller through.

**Common mistake:** scoring 0 because the change is small. Passing a configuration value through to a container is a 2 if that value is a security allow-list.

## 3. Coordinated surfaces

**Question:** How many separate places must change, and stay consistent with each other, for the task to be done?

**Look at:** every surface that exposes or consumes the behaviour: API, command line, user interface, configuration schema, deployment manifests (Docker, Helm, Terraform), documentation, other services and generated clients.

- **0:** One place.
- **1:** Two or three places inside one component that must agree.
- **2:** Several components or layers that must agree, where missing one leaves the feature half-built: for example the API, the CLI and the deployment configuration.

**Common mistake:** counting files. Ten files in one module that change mechanically are a 0 or 1; a two-line change that must also reach a second service is a 2.

## 4. Ambiguity

**Question:** Is there a tempting answer that looks complete but misses what is actually needed?

**Look at:** whether the task describes a symptom or a goal; whether it names the root cause; whether a narrower fix would make the reported symptom disappear while leaving the underlying problem; whether there are two reasonable readings of what to build.

- **0:** One obvious reading, and the obvious fix is the right one.
- **1:** Some interpretation is needed, but the repository makes the intended answer clear once you look.
- **2:** A narrower or different solution would look complete and pass a quick review, yet miss the real problem or the intended design.

**Common mistake:** scoring 0 when the task describes only a symptom. "Session cookies exceed 4096 bytes" invites shrinking one cookie when the real fix may be storing the data server-side.

## 5. Verification

**Question:** If the change were wrong, would anything in the repository catch it?

**Look at:** the tests that cover the code the task touches, what they assert, whether they run locally and in CI, and whether the behaviour can be tested at all without external services, hardware or credentials.

- **0:** Strong tests already cover this behaviour and would fail on a wrong change.
- **1:** Some tests exist but would miss part of what could go wrong.
- **2:** No meaningful tests cover it, or it cannot be tested without systems the repository does not have. A mistake would ship unnoticed.

**Common mistake:** counting test files. A test suite that never exercises this code path is a 2.

## 6. Blast radius

**Question:** If this change shipped wrong, what would it cost, and how easily could it be undone?

**Look at:** who uses the code (only its authors, internal staff, or customers), what the repository map says is production or user-facing, whether it touches security, money or stored data, and whether a mistake could be fixed by reverting a deployment.

- **0:** Throwaway or internal: a prototype, a script, a test fixture, internal documentation. A mistake costs a rerun.
- **1:** Users would notice, but a revert fixes it with no lasting harm.
- **2:** Security, money or data loss, or anything hard to undo: leaked credentials, corrupted records, a bad migration, an authorisation hole.

**Common mistake:** confusing blast radius with difficulty. A trivial change to a password check is a 2 here; a hard change to an internal dashboard is a 0.

## Confidence

After scoring, say how confident you are in the six scores as a whole: **high** when the repository and the task settle every score, **medium** when one or two rest on inference, **low** when you could not find what you needed to judge.
