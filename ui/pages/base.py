from abc import ABC, abstractmethod
import streamlit as st
from typing import Optional
from state import SessionState

class BasePage(ABC):
    def __init__(self, title: str):
        self.title = title
        self.session = SessionState

    @abstractmethod
    def render(self) -> None:
        pass

    def next_step(self) -> None:
        if st.button("Next Step"):
            st.session_state.step += 1
            st.rerun()

    def previous_step(self) -> None:
        if st.button("Previous Step"):
            st.session_state.step -= 1
            st.rerun()

    def show_error(self, message: str) -> None:
        st.error(message)

    def show_success(self, message: str) -> None:
        st.success(message)
