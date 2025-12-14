Raw Datasets (External Sources)

The raw datasets used for this study are large and heterogeneous, and are therefore not stored directly in this repository. Instead, this project references publicly available GitHub repositories that provide real-world API specifications, refactoring histories, and microservice-based systems.

These raw sources were used to construct the curated dataset variants (curated_clean, curated_noisy_light, and curated_noisy_heavy) that are included in this repository and used for all experiments and evaluations.

OpenAPI Specification Corpus

To obtain real-world API specifications and versioned API changes, OpenAPI files were sourced from the following repository:

APIs.guru OpenAPI Directory
https://github.com/APIs-guru/openapi-directory

This repository contains a large collection of publicly available OpenAPI YAML and JSON files. Selected specifications and version pairs were extracted and processed to generate controlled API evolution examples.

OpenRewrite Refactoring Logs

To model structural and semantic API-related refactorings, refactoring histories were sourced from:

OpenRewrite Refactoring Dataset
https://github.com/openrewrite/rewrite

This repository provides real refactoring recipes and change logs, which were analysed and transformed into synthetic API change events to simulate noisy and indirect evolution scenarios.

Spring PetClinic Microservices

To study realistic microservice interactions and downstream dependencies, the following repository was used:

Spring PetClinic Microservices
https://github.com/spring-petclinic/spring-petclinic-microservices

This system provides a distributed microservice architecture with REST APIs and service-to-service dependencies, which was used to create controlled API modifications and dependency graphs.

Purpose of Raw Datasets

The raw datasets listed above were not used directly for training or evaluation. Instead, they were filtered, version-paired, transformed, and annotated to create deterministic and reproducible curated datasets.

Only the curated datasets derived from these sources are included in this repository and referenced throughout the thesis and framework evaluation.
