# Metadata Maintenance

Active metadata builders live in `scripts/data_builders/`.

The `legacy/` directory contains one-shot migration tools for metadata schemas
that are no longer used by the current frameworks. Legacy scripts must not
write to `data/metadata/production/`.
