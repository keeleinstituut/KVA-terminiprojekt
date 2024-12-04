import asyncio
import logging

import panel as pn
import param
from panel.chat import ChatInterface, ChatMessage

import pandas as pd

from app.config import config
from app.models.chatter_model import FilterFactory, Retriever, LLMChat
from utils.db_connection import Connection

import os

pn.extension("perspective")

CONNECTION_PARAMS = {
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
    "db": os.getenv("PG_COLLECTION"),
}

# Configure logging
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)


class FilterActionHandler(param.Parameterized):
    """
    This class is responsible for managing the user interface elements related to filtering
    documents. It also handles the application of filters to the underlying chat model.

    Attributes:
        apply_filters_button (pn.widgets.Button): A button widget to apply the selected filters.
        refresh_choices_button (pn.widgets.Button): A button widget to refresh the options for
            keywords and document titles.
        keyword_selector (pn.widgets.CrossSelector): A widget to select keywords for filtering.
        file_selector (pn.widgets.CrossSelector): A widget to select document titles for filtering.
        limit_slider (pn.widgets.EditableIntSlider): A slider widget to set the limit for the
            number of responses.
        validity_checkbox (pn.widgets.Checkbox): A checkbox widget to filter for documents marked as
            valid by the user previously.
        filterfactory (FilterFactory): An instance of the FilterFactory class used for applying
            filters to the chat model.
        con (Connection): An instance of the Connection class for interacting with Pg database.
        files_df (pd.DataFrame): A DataFrame containing the ids and titles of documents that have finished processing (included in vector database).
        keywords_df (pd.DataFrame): A DataFrame containing the list of keywords.
    """

    def __init__(self, filterfactory, **params):
        self.filterfactory = filterfactory

        self.con = Connection(**CONNECTION_PARAMS)
        self.con.establish_connection()

        self.apply_filters_button = pn.widgets.Button(
            name="Rakenda filtrid",
            button_type="primary",
            width=50,
            margin=(20, 60, 0, 0),
        )

        self.refresh_choices_button = pn.widgets.Button(
            name="Värskenda filtrid",
            button_type="primary",
            width=50,
            margin=(20, 0, 0, 20),
        )

        self.keyword_selector = pn.widgets.CrossSelector(
            name="Märksõnad", value=[], options=[], size=8, width=500
        )
        self.file_selector = pn.widgets.CrossSelector(
            name="Dokumendid", value=[], options=[], size=8, width=500
        )

        self.limit_slider = pn.widgets.EditableIntSlider(
            name="Tekstilõikude arv SKMi sisendis",
            start=1,
            end=20,
            step=1,
            value=5,
            width=500,
        )

        self.validity_checkbox = pn.widgets.Checkbox(
            name="Otsi ainult kehtivatest", width=500
        )

        super().__init__(**params)

        self.refresh_selectors()

    def load_keywords_from_db(self):
        """ Loads the list of keywords from the database. """
        try:
            keywords_df = self.con.statement_to_df(
                """ SELECT DISTINCT keyword FROM keywords ORDER BY keyword"""
            )
            return keywords_df
        except Exception as e:
            logger.error(e)
            return pd.DataFrame(columns=["keyword"])

    def load_files_from_db(self):
        """ Loads the list of processed document titles and ids from the database. """
        try:
            files_df = self.con.statement_to_df(
                """ SELECT id, title FROM documents WHERE current_state = 'uploaded' ORDER BY title"""
            )
            return files_df
        except Exception as e:
            logger.error(e)
            return pd.DataFrame(columns=["id", "title"])

    @param.depends("refresh_choices_button.value", watch=True)
    def refresh_selectors(self):
        """ Refreshes the options for the keyword and document title selectors by utilizing previous methods.
        Responds to clicking refresh_choices_button. """
        logger.info("Refreshing file selection. Established Postgres connection")

        try:
            self.files_df = self.load_files_from_db()
            self.file_selector.options = list(self.files_df["title"])
            self.keywords_df = self.load_keywords_from_db()
            self.keyword_selector.options = list(self.keywords_df["keyword"])
            logger.info("File selection refresh complete.")

        except Exception as e:
            logger.error(e)
            pass

    @param.depends("keyword_selector.value", watch=True)
    def keyword_filtering(self):
        """ Filters the document title options based on the selected keywords. Responds to changes in keyword selector widget. """
        selected_kw_values = self.keyword_selector.value
        if selected_kw_values:
            logger.info(selected_kw_values)
            self.con.establish_connection()
            try:
                result = self.con.execute_sql(
                    """SELECT document_id FROM keywords WHERE keyword IN :kws""",
                    [{"kws": (tuple(selected_kw_values))}],
                )
                compatible_document_ids = [row[0] for row in result["data"]]
                selected_files = list(
                    self.files_df[self.files_df["id"].isin(compatible_document_ids)][
                        "title"
                    ]
                )
                self.file_selector.options = selected_files
                self.file_selector.value = selected_files
            except Exception as e:
                logger.error(e)
        else:
            self.refresh_selectors()

    @param.depends("apply_filters_button.value", watch=True)
    def apply_filters(self):
        """ Applies the selected filters to the chat model using the FilterFactory. Responds to apply_filters_button widget. """
        self.filterfactory.apply_filters(
            files=self.file_selector.value,
            response_limit=self.limit_slider.value,
            document_validity=self.validity_checkbox.value,
        )


def llm_view():
    client_host = os.getenv("QDRANT_HOST")
    client_port = os.getenv("QDRANT_PORT")

    embedding_model = config["embeddings"]["embedding_model"]
    collection_test = os.getenv("QDRANT_COLLECTION")

    api_key = os.getenv("LLM_API_KEY")
    llm_model = config["llm"]["model"]

    filterfactory = FilterFactory()
    retriever = Retriever(
        embedding_model,
        collection_test,
        filterfactory,
        prompt=config["embeddings"]["query_prompt"],
    )
    llm_chatter = LLMChat(llm_model, retriever, api_key)

    retriever.connect(client_host, client_port)
    logger.info("Established Qdrant connection")

    filter_handler = FilterActionHandler(retriever.filterfactory)

    toggle = pn.widgets.ToggleIcon(
        icon="adjustments", active_icon="adjustments-off", size="4em", align="end"
    )

    # Callback to show/hide the filters
    @param.depends(toggle.param.value, watch=True)
    def toggle_filters(show):
        if show:
            filter_column.visible = True
        else:
            filter_column.visible = False

    text_area_input = pn.widgets.TextAreaInput(
        placeholder="Otsi dokumendist", auto_grow=True, max_rows=1
    )

    # Handling sending input and response on Enter press
    def text_area_event_handler(event):
        if event.new.endswith("\n"):  # Check if enter key was pressed
            asyncio.create_task(answer(text_area_input.value_input, text_area_input))

    async def answer(contents, active_widget):
        contents = contents.strip("\n")
        active_widget.param.update({"value": "", "value_input": ""})
        # Empty input field
        ci.send(
            ChatMessage(
                contents,
                user="Terminoloog",
                show_reaction_icons=False,
                show_copy_icon=False,
            ),
            callback=True,
        )

    text_area_input.param.watch(text_area_event_handler, "value_input")

    # Interface assembly
    ci = ChatInterface(
        callback_exception="verbose",
        widgets=text_area_input,
        user="Terminoloog",
        show_send=True,
        show_button_name=False,
        callback=llm_chatter.chat_callback,
        callback_user="Assistent",
        reset_on_send=True,
        show_stop=False,
        show_rerun=False,
        show_undo=False,
        show_copy_icon=False,
        sizing_mode="stretch_width",
        reaction_icons={},
    )

    try:
        filter_column = pn.Column(
            pn.pane.HTML("<label>Vali märksõnad</label>"),
            filter_handler.keyword_selector,
            pn.pane.HTML("<label>Vali dokumendid</label>"),
            filter_handler.file_selector,
            filter_handler.limit_slider,
            filter_handler.validity_checkbox,
            pn.Row(
                filter_handler.apply_filters_button,
                filter_handler.refresh_choices_button,
            ),
            visible=False,
        )

    except Exception as e:
        logger.error(e)

    layout = pn.Row(ci, pn.Column(toggle, filter_column))

    return layout
