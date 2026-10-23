
-- ============================================
-- FACEOPS DATABASE SCHEMA
-- ============================================

-- 1. Class labels
CREATE TABLE IF NOT EXISTS classes (
    label SMALLINT PRIMARY KEY,
    class_name VARCHAR(50) NOT NULL UNIQUE
);

INSERT INTO classes (label, class_name)
VALUES
    (0, 'pawara'),
    (1, 'uchitha'),
    (2, 'dhamith'),
    (3, 'unknown')
ON CONFLICT (label) DO NOTHING;


-- 2. Versioned datasets
CREATE TABLE IF NOT EXISTS dataset_versions (
    id BIGSERIAL PRIMARY KEY,

    raw_dvc_md5 VARCHAR(100) NOT NULL,
    split_manifest_sha256 CHAR(64) NOT NULL,

    total_images INTEGER NOT NULL
        CHECK (total_images > 0),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (raw_dvc_md5, split_manifest_sha256)
);


-- 3. Image metadata
CREATE TABLE IF NOT EXISTS images (
    id BIGSERIAL PRIMARY KEY,

    dataset_version_id BIGINT NOT NULL
        REFERENCES dataset_versions(id)
        ON DELETE CASCADE,

    class_label SMALLINT NOT NULL
        REFERENCES classes(label),

    relative_path TEXT NOT NULL,

    split VARCHAR(10) NOT NULL
        CHECK (split IN ('train', 'val', 'test')),

    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),

    sha256 CHAR(64) NOT NULL,

    UNIQUE (dataset_version_id, relative_path)
);


CREATE INDEX IF NOT EXISTS idx_images_class
ON images(class_label);

CREATE INDEX IF NOT EXISTS idx_images_split
ON images(split);

CREATE INDEX IF NOT EXISTS idx_images_version
ON images(dataset_version_id);
