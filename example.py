"""Example usage of agentspype visualization functionality."""

from typing import Any

from agentspype.agent.agent import Agent
from agentspype.agent.configuration import AgentConfiguration
from agentspype.agent.definition import AgentDefinition
from agentspype.agent.listening import AgentListening
from agentspype.agent.publishing import StateAgentPublishing
from agentspype.agent.state_machine import AgentStateMachine
from agentspype.agent.status import AgentStatus
from agentspype.fsm import State


# Example State Machine
class ExampleStateMachine(AgentStateMachine):
    """Example state machine with custom states and transitions."""

    # Define custom states
    starting = State("Starting", initial=True)
    processing = State("Processing")
    waiting = State("Waiting")
    completed = State("Completed")
    failed = State("Failed")
    end = State("End", final=True)

    # Define transitions
    begin_processing = starting.to(processing)
    wait_for_input = processing.to(waiting)
    resume_processing = waiting.to(processing)
    complete_successfully = processing.to(completed)
    fail_processing = processing.to(failed) | waiting.to(failed)
    stop = starting.to(failed) | processing.to(failed) | waiting.to(failed)
    end = failed.to(end) | completed.to(end)

    def after_transition(self, event: str, state: State) -> None:
        """Handle post-transition actions."""
        if isinstance(self.agent.publishing, StateAgentPublishing):
            self.agent.publishing.publish_transition(event, state)

        # Log the transition
        self.agent.logger().info(f"Transitioned to {state.name} via {event}")


# Example Listening
class ExampleListening(AgentListening):
    """Example listening class with subscription methods."""

    def subscribe(self) -> None:
        """Subscribe to events."""
        self.agent.logger().info("Subscribing to events...")
        # In a real implementation, this would set up actual event subscriptions

    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        self.agent.logger().info("Unsubscribing from events...")
        # In a real implementation, this would clean up event subscriptions

    def handle_external_event(self, event_data: Any) -> None:
        """Example callback for external events."""
        self.agent.logger().info(f"Received external event: {event_data}")
        # This would be called when an external event is received

    def handle_state_change(self, transition_event: Any) -> None:
        """Example callback for state change events."""
        self.agent.logger().info(f"State changed: {transition_event}")
        # This would be called when a state transition event is received


# Example Agent
class ExampleAgent(Agent):
    """Example agent with custom components."""

    definition = AgentDefinition(
        configuration_class=AgentConfiguration,
        events_publishing_class=StateAgentPublishing,
        events_listening_class=ExampleListening,
        state_machine_class=ExampleStateMachine,
        status_class=AgentStatus,
    )

    def initialize(self) -> None:
        """Initialize the agent."""
        super().initialize()
        self.logger().info(f"Initialized {self.__class__.__name__}")


def main() -> None:
    """Demonstrate the visualization functionality."""
    print("🎨 Agent Visualization Demo")
    print("=" * 50)

    # Create an example agent
    agent = ExampleAgent({})

    # Perform some state transitions to make the visualization more interesting
    print("\n📊 Performing state transitions...")
    agent.machine.begin_processing()
    agent.machine.wait_for_input()
    # Current state is now 'waiting'

    print("\n🎯 Creating visualizations...")

    # 1. Create a comprehensive visualization
    print("1. Creating comprehensive agent visualization...")
    agent.visualize(
        save_file=True,
        filename="example_agent_comprehensive",
        show_current_state=True,  # This will highlight the current state
    )

    # 2. Create individual component visualizations
    print("2. Creating state machine visualization...")
    agent.visualize_state_machine(
        save_file=True, filename="example_agent_state_machine"
    )

    print("3. Creating publishing visualization...")
    agent.visualize_publishing(save_file=True, filename="example_agent_publishing")

    print("4. Creating listening visualization...")
    agent.visualize_listening(save_file=True, filename="example_agent_listening")

    # 3. Create all diagrams at once
    print("5. Creating all diagrams at once...")
    all_diagrams = agent.create_all_diagrams(save_files=True, output_dir=".diagrams")

    print(f"\n✅ Created {len(all_diagrams)} diagrams:")
    for name, graph in all_diagrams.items():
        node_count = len(graph.get_node_list())
        edge_count = len(graph.get_edge_list())
        print(f"  - {name}: {node_count} nodes, {edge_count} edges")

    # 4. Demonstrate visualization options
    print("\n🔧 Demonstrating visualization options...")

    # Create a visualization with only state machine
    agent.visualize(
        save_file=True,
        filename="example_agent_state_only",
        include_state_machine=True,
        include_publishing=False,
        include_listening=False,
    )

    # Create a visualization with only publishing and listening
    agent.visualize(
        save_file=True,
        filename="example_agent_events_only",
        include_state_machine=False,
        include_publishing=True,
        include_listening=True,
    )

    print("6. Created state machine only visualization")
    print("7. Created events only visualization")

    # Clean up
    agent.teardown()

    print("\n🎉 Visualization demo completed!")
    print("Check the '.diagrams' directory for the generated PNG files.")
    print("\nYou can now:")
    print("- View the generated diagrams")
    print("- Use agent.visualize() in your own code")
    print("- Customize visualization options")
    print("- Create diagrams for different agent states")


if __name__ == "__main__":
    main()
