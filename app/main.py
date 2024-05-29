import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import panel as pn
from app.views.api_view import api_view
from app.views.file_upload import file_upload

pn.extension()

main_area = pn.Column(pn.pane.Markdown("Vali menüüst lehekülg."))

sidebar_content = pn.Column(
    pn.widgets.Button(name='API päringud', button_type='default'),
    pn.widgets.Button(name='Failide üles laadimine', button_type='default'),
)

def show_api_view(event):
    main_area.clear()
    main_area.append(api_view())

def show_file_upload(event):
    main_area.clear()
    main_area.append(file_upload())

sidebar_content[0].on_click(show_api_view)
sidebar_content[1].on_click(show_file_upload)

template = pn.template.VanillaTemplate(
    title="Tehisintellekti rakendamine riigikaitseterminoloogia valdkonnas",
    sidebar=[sidebar_content],
    main=[main_area],
    sidebar_width=300
)

pn.serve(template, title="Tehisintellekti rakendamine riigikaitseterminoloogia valdkonnas")
