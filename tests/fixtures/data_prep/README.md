# Synthetic data-preparation fixtures

Every row is invented for software verification. No row represents a person,
real trip, station observation or official Taxi Zone geometry.

Trips cover positive/zero tips, cash recording, refunds, invalid denominators,
nonfinite values, missing distance, unknown zones, weather gaps, New York DST
ambiguity/nonexistence, and identical-looking records. `fixture_case` is an
explanation only and is excluded from the content fingerprint.

Money is USD, distance miles, weather temperature degrees Celsius and
precipitation millimeters. Trip timestamps are naive New York time; weather
timestamps contain explicit UTC offsets. Synthetic weather release times are
five minutes after observation, making availability lag visible.

The first two trips have identical fields in the configured fingerprint but
different original record ordinals. They must both survive and share a split.
