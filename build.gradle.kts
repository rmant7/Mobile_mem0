plugins {
    kotlin("jvm") version "2.1.0"
    kotlin("plugin.serialization") version "2.1.0"
    `java-library`
    `maven-publish`
}

group = "com.github.rmant7"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(21)
    // Java 17 bytecode, not 21: this is meant to be usable from an Android
    // app, and D8 (Android's dexer) is the real constraint on how new the
    // emitted class files can be, regardless of which JDK compiles them.
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(17)
}

tasks.test {
    useJUnitPlatform()
}

java {
    withSourcesJar()
}

// Lets JitPack (and anyone else) resolve this repo as a Maven dependency
// with no further setup — JitPack builds any tagged commit by running the
// Gradle publish tasks a project already declares.
publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
        }
    }
}
