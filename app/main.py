"""
Main Panel application - session handling only.
Heavy initialization is done once in init_app.py
"""
import panel as pn

# Import from cached init module (runs only ONCE)
from init_app import logger, file_upload, llm_view, prompt_view, document_management


# Define the layout areas for the Panel UI
def create_file_upload_area():
    return pn.Column(file_upload())


def create_chat_view_area():
    return pn.Column(llm_view())


def create_prompt_view_area():
    return pn.Column(prompt_view())


def create_document_management_area():
    return pn.Column(document_management())


# Create the template and structure for the UI
def create_template(guest_mode=False):
    chat_view_area = create_chat_view_area()

    tabs = pn.Tabs(
        ("Dokumendiotsing", chat_view_area),
    )

    if not guest_mode:
        # Only create these views for non-guest users
        document_management_area = create_document_management_area()
        file_upload_area = create_file_upload_area()
        prompt_view_area = create_prompt_view_area()
        
        tabs.append(("Dokumentide haldamine", document_management_area))
        tabs.append(("Failide üleslaadimine", file_upload_area))
        tabs.append(("Promptide haldamine", prompt_view_area))

    template = pn.template.VanillaTemplate(
        title="Tehisintellekti rakendamine riigikaitseterminoloogia valdkonnas",
        main=[tabs],
        sidebar_width=300,
    )

    return template


# Main function to create and serve the Panel app
def create_app(guest_mode: bool = False):
    return create_template(guest_mode=guest_mode)


# Session handling
username = pn.state.user
logger.info(f"New session: user={username}")

if username == "kylaline":
    template = create_app(guest_mode=True).servable()
else:
    template = create_app().servable()
