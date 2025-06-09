from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from manufacturing_agent.tools.file_tools import WriteFileTool, ReadFileTool

# Initialize the tools
write_tool = WriteFileTool()
read_tool = ReadFileTool()

@CrewBase
class ManufacturingAgentCrew():
	"""ManufacturingAgent crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def manufacturing_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['manufacturing_analyst'],
			tools=[read_tool],
			verbose=True
		)

	@agent
	def decision_maker(self) -> Agent:
		return Agent(
			config=self.agents_config['decision_maker'],
			verbose=True
		)

	@agent
	def output_generator(self) -> Agent:
		return Agent(
			config=self.agents_config['output_generator'],
			tools=[write_tool],
			verbose=True
		)

	@task
	def analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['analysis_task'],
			agent=self.manufacturing_analyst()
		)

	@task
	def decision_task(self) -> Task:
		return Task(
			config=self.tasks_config['decision_task'],
			agent=self.decision_maker(),
			context=[self.analysis_task()]
		)

	@task
	def output_task(self) -> Task:
		return Task(
			config=self.tasks_config['output_task'],
			agent=self.output_generator(),
			context=[self.decision_task()]
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the ManufacturingAgent crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)