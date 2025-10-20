import logging
from rich.table import Table
from rich.console import Console
from adet.fluid.node import FlowNode


logger = logging.getLogger(__name__)


def tabulate_components(flow_sequence):
    """
    Print the component sequence as a rich table.
    """
    table = Table()
    table.add_column('Index', justify='right', no_wrap=True)
    table.add_column('Component')

    for index, instance in enumerate(flow_sequence):
        component_name = str(instance).partition(' ')[0]
        style = 'blue' if isinstance(instance, FlowNode) else 'green'
        table.add_row(str(index), component_name, style=style)

    Console().print(table)
