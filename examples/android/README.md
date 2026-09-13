# Android example

`ChatMemoryViewModel.kt` is a realistic, illustrative usage example — how
this library actually gets wired into an Android chat screen — not a
buildable Gradle module. Mobile mem0 itself has zero Android dependency (it
only needs `context.filesDir` as a plain `java.io.File`, passed in by the
caller), so there's nothing Android-specific to build or test here; this
file exists to show the *shape* of real usage, copy-pasteable into your own
`ViewModel`.

It assumes:
- a `MyLlmRuntime` you already have for generating text (any local or
  remote model call with a `suspend fun generate(prompt: String): String`
  shape works as a [`MemoryExtractor`](../../src/main/kotlin/ai/localstudio/memory/MemoryProvider.kt));
- standard AndroidX `ViewModel` / `viewModelScope` for lifecycle-aware
  coroutines.

See the root [README](../../README.md#quick-start) for the plain-Kotlin
Quick Start this example builds on.
