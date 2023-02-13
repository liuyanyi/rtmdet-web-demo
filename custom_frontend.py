import importlib

import streamlit as st
from pinferencia.frontend.api_manager import APISet
from pinferencia.frontend.app import Server
from pinferencia.frontend.config import (DEFAULT_DETAIL_DESCRIPTION,
                                         DEFAULT_SHORT_DESCRIPTION)
from template.rtmdet_template import Template as RTMDET_TEMPLATE


class CustomModelManager:
    # TODO: select API set automatically
    def __init__(self, backend_server: str, api_set: set = APISet.DEFAULT):
        api_manager_module = importlib.import_module(
            f".frontend.api_manager.{api_set}",
            package="pinferencia",
        )
        self.api_manager = api_manager_module.APIManager(server=backend_server)

    # @st.cache()
    def list(self, model_name: str = None):
        response_json = self.api_manager.list(model_name=model_name)
        if not isinstance(response_json, list):
            raise Exception(response_json)
        return response_json

    def predict(
        self,
        model_name: str,
        data: object,
        version_name: str = None,
        parse_data: bool = True,
    ) -> object:
        return self.api_manager.predict(
            model_name=model_name,
            data=data,
            version_name=version_name,
            parse_data=parse_data,
        )


class CustomServer(Server):

    def __init__(
        self,
        backend_server: str,
        title: str = None,
        short_description: str = DEFAULT_SHORT_DESCRIPTION,
        detail_description: str = DEFAULT_DETAIL_DESCRIPTION,
        api_set: set = APISet.DEFAULT,
    ):
        """Frontend Server

        Args:
            backend_server (str): backend server address
            title (str, optional): title of the app. Defaults to None.
            short_description (str, optional):
                short description. Defaults to None.
            detail_description (str, optional):
                detailed description. Defaults to None.
        """
        self.backend_server = backend_server
        self.title = title if title else "Pinferencia"
        self.short_description = short_description
        self.detail_description = detail_description
        self.model_manager = CustomModelManager(
            backend_server=self.backend_server,
            api_set=api_set,
        )
        self.custom_templates = {}
        self.render()
        hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            footer:after {
                content:'Made with Pinferencia and Streamlit';
                visibility: visible;
                display: block;
            }
            </style>
            """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    def get_task_options(self) -> dict:
        task_options = dict()
        task_options['rtmdet_template'] = RTMDET_TEMPLATE
        return task_options

    def render(self):
        """Render the page"""
        # prepare upper right corner menu items and page configs
        menu_items = {
            "Get Help": "https://github.com/underneathall/pinferencia/issues",  # noqa
            "Report a bug": "https://github.com/underneathall/pinferencia/issues",  # noqa
            "About": self.detail_description,
        }

        st.set_page_config(
            layout="centered",
            initial_sidebar_state="auto",
            page_title=self.title,
            menu_items=menu_items,
        )

        # render sidebar header
        st.sidebar.title(self.title)
        st.sidebar.markdown(self.short_description)

        # retrieve models from backend
        models = self.get_models()

        # render the model select box
        model_name = st.sidebar.selectbox(
            label="Select the Model",
            options=models,
        )

        # retrieve model version metadata from backend
        versions = (
            {v["name"]: v for v in self.model_manager.list(model_name)}
            if model_name
            else {}
        )

        # render the version select box
        version_name = st.sidebar.selectbox(
            label="Select the Version",
            options=versions,
        )

        # load built-in and custom task options
        avia_tasks = self.get_task_options()
        task_options = list(avia_tasks.keys())

        # if a version is selected
        if version_name:
            # try to read the task of this model version
            version_task = versions[version_name].get("task")
            select_box_kwargs = {
                "label": "Select the Task",
                "options": task_options,
                "format_func": lambda x: x.replace("_", " ").title(),
            }

            # if the task of this model version is defined,
            # select it by default.
            # otherwise, select the first task
            if version_task:
                try:
                    select_box_kwargs["index"] = task_options.index(version_task)
                except ValueError:
                    pass
            task = st.sidebar.selectbox(**select_box_kwargs)

            # if debug mode is enabled, set the api manager to debug mode
            if st.sidebar.checkbox("Debug"):
                self.model_manager.api_manager.debug = True

            # first looking for the selected task type in custom templates,
            # then the built-in templates
            tmpl_cls = avia_tasks[task]

            # initialize the template and render
            tmpl = tmpl_cls(
                model_name=model_name,
                version_name=version_name,
                model_manager=self.model_manager,
                metadata=versions[version_name],
            )
            tmpl.render()


service = CustomServer(
    title="RTMDet Demo", # 
    short_description="This is the short description", # 
    backend_server="http://127.0.0.1:8000",
)