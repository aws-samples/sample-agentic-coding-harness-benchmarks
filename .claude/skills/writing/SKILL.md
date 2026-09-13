---
name: writing
description: Write prose people will actually read. Use for any prose you produce - docs, READMEs, release notes, blog posts, PR descriptions, issue text, commit bodies, design docs, emails, chat answers. Applies Orwell's six rules and strips the LLM tells - passive voice, dead metaphors, and -ly padding.
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.0"
---

# Writing Skill

If you want people to read your stuff, follow Orwell's rules.

LLMs are not your friends. They write in passive voice, mix figures of speech that do not fit together, and pad every sentence with -ly words. Cut all of it.

## Orwell's rules

1. Never use a metaphor, simile, or other figure of speech which you are used to seeing in print.
2. Never use a long word where a short one will do.
3. If it is possible to cut a word out, always cut it out.
4. Never use the passive where you can use the active.
5. Never use a foreign phrase, a scientific word, or a jargon word if you can think of an everyday English equivalent.
6. Break any of these rules sooner than say anything outright barbarous.

Rule 6 outranks the other five. A stiff sentence that obeys rules 1-5 is worse than a plain one that breaks one of them.

## How to apply each rule

### 1. No stock figures of speech

Kill any phrase you have read a hundred times. If two images sit in one sentence, they will clash and the reader sees nothing.

Ban list, not exhaustive: game changer, at the end of the day, low-hanging fruit, moving the needle, paradigm shift, deep dive, unlock value, seamless, robust, journey, landscape, ecosystem (unless you mean living things), leverage as a verb, delve, tapestry, testament to, navigate the complexities.

Use a plain statement, or invent an image that fits the thing you are describing.

- Bad: This release is a game changer that unlocks seamless value across the ecosystem.
- Good: This release cuts registry startup from 40 seconds to 3.

### 2. Short words

Prefer the short word every time: use not utilize, help not facilitate, start not commence, about not approximately, use not leverage, get not obtain, show not demonstrate, need not necessitate, before not prior to, after not subsequent to, so not accordingly, most not the majority of, can not possesses the capability to.

### 3. Cut words

Delete every word that carries no weight. Common padding: in order to, the fact that, it should be noted that, it is important to note, as previously mentioned, in terms of, with respect to, a number of, at this point in time, basically, essentially, actually, really, very, quite, simply, just.

Delete throat-clearing openers. Start with the fact.

- Bad: It is important to note that, in terms of performance, the cache basically helps quite a lot.
- Good: The cache cuts p99 latency by half.

Aim to cut a first draft by a third. Then read it again and cut more.

### 4. Active voice

Name the actor, then the verb. Passive voice hides who did what, and hidden actors are how bugs and bad decisions escape review.

- Bad: The token is validated by the auth service and errors are logged.
- Good: The auth service validates the token and logs errors.

Hunt for is/are/was/were/been/being next to a past participle. Also hunt "there is", "there are", and "it is" - each one usually buries the real subject.

Keep the passive only when the actor is unknown or truly beside the point: "The row was deleted at some point before the migration."

### 5. Everyday English

Say the thing in words a competent reader knows. Drop the Latin and the jargon: use for example not e.g., that is not i.e., by itself not per se, the opposite not vice versa, so far not to date, roughly not circa.

Keep the technical term when it is the precise name of the thing. `JWT`, `Fargate`, `idempotent`, and `race condition` earn their place. `synergy`, `holistic`, and `operationalize` do not.

Write out an acronym on first use, then use it.

### 6. Sound like a person

Read the sentence aloud. If no one would say it, rewrite it. Break any rule above rather than write something ugly, stilted, or unclear.

## Extra rules for LLM prose

- Cut -ly adverbs. Pick a stronger verb instead. `significantly improved` -> `doubled`. `carefully validates` -> `validates`.
- No "not only X but also Y". No "X isn't just Y - it's Z". No em-dash reveal at the end of a sentence.
- No lists of three when two facts will do.
- No summary paragraph that repeats what you just said.
- No praise of the reader, the code, or yourself. No "great question", no "powerful and flexible".
- One idea per sentence. Short sentences beat long ones with semicolons.
- Concrete over abstract: exact numbers, file paths, symbol names, commands.
- Say what changed and what breaks. Skip the vision.
- Do not hedge twice. "may possibly" -> "may". Pick one level of certainty and own it.
- Use present tense for how the system behaves, past tense for what you did.
- Never open with "In today's fast-paced world" or any variant.

## Sentence-shape tells

Word-level fixes are not enough. LLMs lean on a handful of sentence shapes that read as machine-made even when every word is plain. Kill these.

- No antithesis. Do not pair "X, but Y" or "not X, rather Y" for rhythm. Say the one thing you mean.
  - Bad: The cache is not a workaround, it is the design.
  - Good: The cache is the design.
- No corrective negation. Do not define a thing by first saying what it is not.
  - Bad: This isn't about speed, it's about correctness.
  - Good: This fixes a correctness bug.
- No contrasting pairs or negative parallelism. Drop the "not just X, but Y" and "less A, more B" frames.
  - Bad: We didn't add a feature, we removed a footgun.
  - Good: We removed the retry loop that double-charged users.
- No negative anaphora. Do not open three sentences in a row with "No..." or "Never..." for effect. (This list is a list, not prose.)
- No setup/payoff or landing sentences. Do not build a sentence whose only job is to tee up the next one, and do not end a paragraph on a short punchy line meant to resonate.
  - Bad: There was one thing left to fix. The timeout.
  - Good: The last fix was the 30-second timeout.
- No parataxis for drama. Do not stack short clauses to build rhythm ("It compiles. It ships. It works.").
- No parallel sentence structures within a paragraph. If two sentences share the same skeleton, rewrite one.
- No paragraph pinning. Do not top and tail a paragraph with the same idea to frame it.
- No summary beats. Do not restate the point you just made in different words.
- No stacked noun phrases. Break "a cloud-native observability data ingestion pipeline" into words that do work.
- No nominalization. Turn the noun back into its verb: "perform a validation of" -> "validate", "make a decision" -> "decide", "provide support for" -> "support".
- Vary sentence length on purpose, not on a pattern. Mix short and long so the rhythm is unpredictable. Do not alternate long-short-long-short.

## Revision pass

Run this on every draft before you ship it:

1. Read it aloud. Fix anything you stumble on.
2. Search for is/are/was/were + participle. Flip each to active or justify it.
3. Search for `ly ` and delete or replace each hit.
4. Delete every phrase from the ban lists above.
5. Cut the first sentence of each paragraph if the second one already says it.
6. Scan for the sentence-shape tells above: antithesis, corrective negation, contrasting pairs, setup/payoff, parallel structure, stacked nouns. Rewrite each into a plain statement.
7. Count words. Cut a third.
8. Check every claim against something real - a file, a command output, a number.
9. Apply rule 6 last: read once more and undo anything that now sounds wrong.

## Example

Before:

> It should be noted that a significant number of performance improvements have been implemented in this release, which fundamentally transforms the observability landscape by leveraging a robust new telemetry pipeline that seamlessly facilitates the collection of metrics at scale.

After:

> This release adds a telemetry pipeline. The collector batches metrics every 10 seconds and cuts registry CPU use by 30%.

62 words to 22. Named the actor. Gave the numbers.
