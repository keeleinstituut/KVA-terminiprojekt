import sys
import os
import panel as pn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.views.api_view import api_view
from app.views.file_upload import file_upload

file_upload_area = pn.Column(file_upload())
api_view_area = pn.Column(api_view())

tabs = pn.Tabs(
    ('Failide üles laadimine', file_upload_area),
    ('API päringud', api_view_area),
)

template = pn.template.VanillaTemplate(
    title="Tehisintellekti rakendamine riigikaitseterminoloogia valdkonnas",
    main=[tabs],
    sidebar_width=300
)

pn.serve(template, title="Tehisintellekti rakendamine riigikaitseterminoloogia valdkonnas")
