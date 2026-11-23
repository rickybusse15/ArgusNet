# Site memory (F6, first slice)

**Status:** shipped, opt in. `rust/argusnet-sim/src/site_memory.rs`;
`argusnet-rs sim --site-memory DIR --site ID`, and `argusnet-rs site`.

A mission's map is kept as a **revision** of its site: an immutable document named by
the BLAKE2b hash of its content, holding each 10 m cell's ground and its claimed
sigma, where its returns are placed (one sigma), a structure's top, the returns
behind it, the mission that made it and the revision it followed. Revisions are only
added, in order, and only in the site's own coordinate frame. A file whose content is
not what its name says is refused.

The next mission is compared with the last, and what differs is kept beside it.

| Answer | When |
| --- | --- |
| Change | Both missions knew the cell; the difference passes 4 sigma of both claims and 1 m (ground) or 2 m (structure); at least two touching cells |
| No change | Within that |
| Undetermined | The two placed sigmas together pass half a cell, or one saw a structure and the other had fewer than 20 ground returns to say it was not there |
| Known to one only | Not compared |

A structure in a cell or beside it is there: its edge is seen from different sides, and
a roof in part. Ground beside a structure either mission knows is part wall, and not
judged.

```sh
argusnet-rs sim --config site.json --profile field_v3 --replay out.json \
    --site-memory sites --site montague
argusnet-rs site --memory sites --site montague --output changes.json
```

## Measured (2026-11-23)

`field_v3`, both sites, four missions each on the unchanged world (seeds 7, 42, 4451,
1010), then one (707) on a world with 8 buildings of 150–900 m² taken down and 8 of
480 m² put up in open places.

| | West Pleasant View | Montague City |
| --- | --- | --- |
| False changes, three comparisons of the unchanged world (about 15,100 cells each) | 0, 0, 0 | 0, 0, 1 |
| Undetermined cells | 4, 7, 4 | 1, 2, 2 |
| Put up, found | 8 of 8 | 8 of 8 |
| Taken down, found | 7 of 8 | 6 of 8 |
| Changes found that are not there, on the changed world | 0 | 0 |

The three missed are three of the smallest (152–172 m², under two cells). The first
version claimed 8 changes in four comparisons that were not there: structures in a
corner of a cell, missed by a thin scatter or seen in part. The two rules above came
of it, on seeds 7, 42 and 4451; 1010 and the changed worlds were run after.

## What it does not do

- A mission does not *use* the last revision: nothing is planned or flown on it. What
  is stale and what is seen now would have to be told apart first.
- Wires, poles and images are not remembered.
- A grid is laid from the map's origin, so a site's missions must share one.
- Alignment is the navigation's own: two maps are not registered to each other.
- Both sites' worlds are simulated, and the sensors' numbers assumed.

## Tests

`site_memory.rs` (12) and `rust/argusnet-cli/tests/site_memory.rs` (two missions of one
site through the CLI, and what is refused).
