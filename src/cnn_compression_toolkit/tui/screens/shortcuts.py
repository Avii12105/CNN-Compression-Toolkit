"""Shortcuts help screen — displays all keyboard shortcuts."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, Static


class ShortcutsScreen(Screen):
    """Display all keyboard shortcuts for the toolkit."""

    BINDINGS = [
        ("escape", "pop_screen", "Close"),
        ("q", "pop_screen", "Close"),
        ("question_mark", "pop_screen", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="shortcuts-scroll"):
            with Vertical(id="shortcuts-screen"):
                yield Label("Keyboard Shortcuts", classes="section-title")
                yield Static("Press ? or Escape to close", classes="subsection-title")

                # Global shortcuts
                yield Static("Global", classes="subsection-title")
                yield DataTable(id="global-table", zebra_stripes=True)

                # Current screen shortcuts (filled dynamically)
                yield Static("Current Screen", classes="subsection-title")
                yield DataTable(id="screen-table", zebra_stripes=True)

                yield Button("Close", id="btn-close", variant="primary")

    def on_mount(self) -> None:
        """Populate shortcut tables."""

        # Global shortcuts table
        global_table = self.query_one("#global-table", DataTable)
        global_table.add_columns("Key", "Action")
        global_shortcuts = [
            ("?", "Toggle this help screen"),
            ("q", "Quit the toolkit"),
            ("Ctrl+Q", "Quit the toolkit"),
            ("Tab", "Switch between screens"),
            ("Esc", "Close current screen / help"),
        ]
        for key, action in global_shortcuts:
            global_table.add_row(key, action)

        # Current screen shortcuts
        screen_table = self.query_one("#screen-table", DataTable)
        screen_table.add_columns("Key", "Action")

        screen = self.app.screen
        bindings = getattr(screen, "BINDINGS", [])
        if bindings:
            for binding in bindings:
                key = binding[0] if isinstance(binding, tuple) else str(binding)
                action = binding[1] if isinstance(binding, tuple) and len(binding) > 1 else ""
                description = (
                    binding[2]
                    if isinstance(binding, tuple) and len(binding) > 2
                    else action
                )
                screen_table.add_row(key, description)
        else:
            screen_table.add_row("None", "No screen-specific shortcuts")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""

        if event.button.id == "btn-close":
            self.app.pop_screen()
            self.app._shortcuts_visible = False
