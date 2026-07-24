package migrate

import "testing"

func TestRestoreGroupItemChannelModelIndex(t *testing.T) {
	db := openMigrationTestDB(t)

	statements := []string{
		"CREATE TABLE group_items (id INTEGER PRIMARY KEY, group_id INTEGER NOT NULL, channel_id INTEGER NOT NULL, channel_key_id INTEGER NOT NULL DEFAULT 0, model_name TEXT NOT NULL, priority INTEGER, weight INTEGER)",
		"CREATE UNIQUE INDEX idx_group_channel_model ON group_items(group_id, channel_id, channel_key_id, model_name)",
		"INSERT INTO group_items (id, group_id, channel_id, channel_key_id, model_name, priority, weight) VALUES (10, 1, 20, 0, 'grok-4.5', 1, 100)",
		"INSERT INTO group_items (id, group_id, channel_id, channel_key_id, model_name, priority, weight) VALUES (11, 1, 20, 101, 'grok-4.5', 2, 50)",
		"INSERT INTO group_items (id, group_id, channel_id, channel_key_id, model_name, priority, weight) VALUES (12, 1, 20, 102, 'grok-4.5', 3, 25)",
	}
	for _, statement := range statements {
		if err := db.Exec(statement).Error; err != nil {
			t.Fatalf("seed migration test db failed: %v", err)
		}
	}

	for attempt := 0; attempt < 2; attempt++ {
		if err := restoreGroupItemChannelModelIndex(db); err != nil {
			t.Fatalf("restoreGroupItemChannelModelIndex attempt %d returned error: %v", attempt+1, err)
		}
	}

	type savedRow struct {
		ID       int
		Priority int
		Weight   int
	}
	var rows []savedRow
	if err := db.Table("group_items").
		Select("id", "priority", "weight").
		Where("group_id = ? AND channel_id = ? AND model_name = ?", 1, 20, "grok-4.5").
		Find(&rows).Error; err != nil {
		t.Fatalf("load migrated group items: %v", err)
	}
	if len(rows) != 1 || rows[0].ID != 10 || rows[0].Priority != 1 || rows[0].Weight != 100 {
		t.Fatalf("expected oldest group item to be preserved, got %#v", rows)
	}

	err := db.Exec("INSERT INTO group_items (group_id, channel_id, channel_key_id, model_name, priority, weight) VALUES (1, 20, 999, 'grok-4.5', 9, 9)").Error
	if err == nil {
		t.Fatal("expected restored three-column unique index to reject another channel-key row")
	}
}
