# Requirement: checkpoint respawn must not softlock progression

ID: checkpoint_no_softlock

After a player collects the keycard, opens the security door, and activates the
checkpoint beyond that door, death and respawn at the checkpoint must preserve
the opened-door progression state. The player must still be able to reach
extraction after respawn. A closed security door after checkpoint respawn is a
progression softlock.
