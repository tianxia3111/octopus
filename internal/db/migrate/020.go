package migrate

import (
	"fmt"

	"github.com/bestruirui/octopus/internal/model"
	"gorm.io/gorm"
)

func init() {
	RegisterBeforeAutoMigration(Migration{
		Version: 20,
		Up:      restoreGroupItemChannelModelIndex,
	})
}

func restoreGroupItemChannelModelIndex(db *gorm.DB) error {
	if db == nil {
		return fmt.Errorf("db is nil")
	}
	if !db.Migrator().HasTable(&model.GroupItem{}) {
		return nil
	}

	return db.Transaction(func(tx *gorm.DB) error {
		if tx.Migrator().HasIndex(&model.GroupItem{}, "idx_group_channel_model") {
			if err := tx.Migrator().DropIndex(&model.GroupItem{}, "idx_group_channel_model"); err != nil {
				return fmt.Errorf("drop group item channel-model index: %w", err)
			}
		}

		// The removed channel-key routing feature allowed multiple rows for the
		// same group/channel/model tuple. Keep the oldest row so the original
		// three-column uniqueness constraint can be restored deterministically.
		if err := tx.Exec(`
			DELETE FROM group_items
			WHERE id NOT IN (
				SELECT id FROM (
					SELECT MIN(id) AS id
					FROM group_items
					GROUP BY group_id, channel_id, model_name
				)
			)
		`).Error; err != nil {
			return fmt.Errorf("deduplicate group items: %w", err)
		}

		if err := tx.Migrator().CreateIndex(&model.GroupItem{}, "idx_group_channel_model"); err != nil {
			return fmt.Errorf("restore group item channel-model index: %w", err)
		}
		return nil
	})
}
