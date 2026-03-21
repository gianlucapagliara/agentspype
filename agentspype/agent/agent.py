import logging
from typing import TYPE_CHECKING, Any

from agentspype.agency import Agency
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition

if TYPE_CHECKING:
    import pydot

    from agentspype.agent.listening import AgentListening
    from agentspype.agent.publishing import AgentPublishing
    from agentspype.agent.state_machine import AgentStateMachine
    from agentspype.agent.status import AgentStatus
    from agentspype.visualization.agent_visualization import AgentVisualization


class Agent:
    _logger: logging.Logger | None = None

    # === Definition ===

    definition: AgentDefinition

    # === Initialization ===

    def __init__(
        self,
        configuration: AgentConfiguration | dict[str, Any],
        parent_id: int | None = None,
    ):
        self._configuration = (
            configuration
            if isinstance(configuration, AgentConfiguration)
            else self.definition.configuration_class(**configuration)
        )

        self._events_publishing = self.definition.events_publishing_class(self)
        self._events_listening = self.definition.events_listening_class(self)
        self._state_machine = self.definition.state_machine_class(self)
        self._status = self.definition.status_class()

        self._parent_id: int | None = parent_id
        self._base_name = self.__class__.__name__

        Agency.register_agent(self)

        self.initialize()

    def initialize(self) -> None:
        pass

    def teardown(self) -> None:
        self.listening.unsubscribe()
        Agency.deregister_agent(self)

    def clone(self) -> "Agent":
        return self.__class__(self.configuration)

    def __del__(self) -> None:
        try:
            self.teardown()
        except Exception:
            pass

    # === Class Methods ===

    @classmethod
    def logger(cls) -> logging.Logger:
        if cls._logger is None:
            cls._logger = logging.getLogger(cls.__name__)
        return cls._logger

    # === Properties ===

    @property
    def name(self) -> str:
        return f"AID:{id(self)}|{self._base_name}"

    @property
    def complete_name(self) -> str:
        if self._parent_id is None:
            return f"PID:-|{self.name}"
        return f"PID:{self.parent_id}|{self.name}"

    @property
    def configuration(self) -> AgentConfiguration:
        return self._configuration

    @property
    def machine(self) -> "AgentStateMachine":
        return self._state_machine

    @property
    def listening(self) -> "AgentListening":
        return self._events_listening

    @property
    def publishing(self) -> "AgentPublishing":
        return self._events_publishing

    @property
    def status(self) -> "AgentStatus":
        return self._status

    @property
    def parent_id(self) -> int | None:
        return self._parent_id

    @parent_id.setter
    def parent_id(self, value: int | None) -> None:
        if value == self._parent_id:
            return

        if self._parent_id is not None:
            raise ValueError("Parent ID is already set")

        self._parent_id = value

    # === Components ===

    def get_components(self) -> list[Any]:
        """Return a list of sub-components for this agent.

        Override this method in subclasses to expose domain-specific components
        (e.g. OrderComponent, TransferComponent) that will appear in the
        agent's visualization diagram.

        Each component should ideally have a ``name`` attribute; otherwise its
        class name will be used as the label.

        Returns:
            list[Any]: A list of component objects. Empty by default.
        """
        return []

    # === Visualization Methods ===

    def _get_visualizer(self) -> "AgentVisualization":
        """Get the agent visualizer (lazy loaded)."""
        try:
            from agentspype.visualization.agent_visualization import AgentVisualization

            return AgentVisualization()
        except ImportError as e:
            raise ImportError(
                "Visualization dependencies not available. "
                "Please ensure pydot is installed."
            ) from e

    def visualize(
        self,
        save_file: bool = False,
        filename: str | None = None,
        output_dir: str = ".diagrams",
        include_state_machine: bool = True,
        include_publishing: bool = True,
        include_listening: bool = True,
        include_components: bool = True,
        show_current_state: bool = True,
        edge_style_map: dict[str, dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> "pydot.Dot":
        """Create a comprehensive visualization of the agent.

        Args:
            save_file: Whether to save the diagram to a file
            filename: Custom filename for the saved diagram
            output_dir: Directory to save the diagram
            include_state_machine: Whether to include state machine visualization
            include_publishing: Whether to include publishing visualization
            include_listening: Whether to include listening visualization
            include_components: Whether to include agent sub-components
            show_current_state: Whether to highlight the current state
            edge_style_map: Custom edge styling for state machine transitions.
                Maps transition event names to style dicts with keys like
                "color", "style", "penwidth".
            **kwargs: Additional arguments passed to visualization components

        Returns:
            pydot.Dot: The generated diagram
        """
        visualizer = self._get_visualizer()
        return visualizer.visualize(
            self,
            save_file=save_file,
            filename=filename or f"{self.__class__.__name__}_agent",
            output_dir=output_dir,
            include_state_machine=include_state_machine,
            include_publishing=include_publishing,
            include_listening=include_listening,
            include_components=include_components,
            show_current_state=show_current_state,
            edge_style_map=edge_style_map,
            **kwargs,
        )

    def visualize_state_machine(
        self,
        save_file: bool = False,
        filename: str | None = None,
        output_dir: str = ".diagrams",
        **kwargs: Any,
    ) -> "pydot.Dot":
        """Create a visualization of the agent's state machine.

        Args:
            save_file: Whether to save the diagram to a file
            filename: Custom filename for the saved diagram
            output_dir: Directory to save the diagram
            **kwargs: Additional arguments passed to state machine visualization

        Returns:
            pydot.Dot: The generated state machine diagram
        """
        visualizer = self._get_visualizer()
        return visualizer.visualize_state_machine_only(
            self,
            save_file=save_file,
            filename=filename,
            output_dir=output_dir,
            **kwargs,
        )

    def visualize_publishing(
        self,
        save_file: bool = False,
        filename: str | None = None,
        output_dir: str = ".diagrams",
        **kwargs: Any,
    ) -> "pydot.Dot":
        """Create a visualization of the agent's event publishing.

        Args:
            save_file: Whether to save the diagram to a file
            filename: Custom filename for the saved diagram
            output_dir: Directory to save the diagram
            **kwargs: Additional arguments passed to publishing visualization

        Returns:
            pydot.Dot: The generated publishing diagram
        """
        visualizer = self._get_visualizer()
        return visualizer.visualize_publishing_only(
            self,
            save_file=save_file,
            filename=filename,
            output_dir=output_dir,
            **kwargs,
        )

    def visualize_listening(
        self,
        save_file: bool = False,
        filename: str | None = None,
        output_dir: str = ".diagrams",
        **kwargs: Any,
    ) -> "pydot.Dot":
        """Create a visualization of the agent's event listening.

        Args:
            save_file: Whether to save the diagram to a file
            filename: Custom filename for the saved diagram
            output_dir: Directory to save the diagram
            **kwargs: Additional arguments passed to listening visualization

        Returns:
            pydot.Dot: The generated listening diagram
        """
        visualizer = self._get_visualizer()
        return visualizer.visualize_listening_only(
            self,
            save_file=save_file,
            filename=filename,
            output_dir=output_dir,
            **kwargs,
        )

    def create_all_diagrams(
        self, save_files: bool = True, output_dir: str = ".diagrams", **kwargs: Any
    ) -> dict[str, "pydot.Dot"]:
        """Create all possible diagrams for the agent.

        Args:
            save_files: Whether to save all diagrams to files
            output_dir: Directory to save the diagrams
            **kwargs: Additional arguments passed to visualization components

        Returns:
            Dict[str, pydot.Dot]: Dictionary mapping diagram names to pydot.Dot objects
        """
        visualizer = self._get_visualizer()
        return visualizer.create_component_diagrams(
            self, save_files=save_files, output_dir=output_dir, **kwargs
        )
