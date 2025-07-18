from datetime import timedelta

from feast import BigQuerySource, Entity, Feature, FeatureView, ValueType, Field
from feast.types import Float32, Int64 

sample = Entity(name="sample_id", join_keys=["sample_id"])

iris_source = BigQuerySource(
    table="sharp-cursor-461116-t1.iris_mock_practice.features_table",
    timestamp_field="event_timestamp"
)

iris_features = FeatureView(
    name="iris_features",
    entities=[sample],
    ttl=timedelta(days=30),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_source,
    tags={"team":"feature_store"}
)