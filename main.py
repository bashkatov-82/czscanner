# main.py - mobile Crypto Zone Scanner (Kivy, Android)
import os
import sys
import threading
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from kivy.app import App
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner

import crypto_zone_screener as czs
import fast_scan

try:
    import ai_agents
    HAS_AGENTS = True
except Exception:
    HAS_AGENTS = False
try:
    import deep_analysis
    HAS_DEEP = True
except Exception:
    HAS_DEEP = False
try:
    import chart_ai
    HAS_COUNCIL = True
except Exception:
    HAS_COUNCIL = False
try:
    import brain
    HAS_BRAIN = True
except Exception:
    HAS_BRAIN = False


def make_args(interval, top):
    return SimpleNamespace(
        mode="scan", symbol="BTCUSDT", interval=interval, limit=300,
        top=top, display=30, min_quote=10_000_000,
        accum_threshold=3.0, distrib_threshold=3.0, min_zone=3, zone_age=5,
        bins=60, stop=3.0, take=6.0, hold=30, backtest_top=10, sleep=0.12, loop=0)


class DetailScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        root = BoxLayout(orientation="vertical")
        top = BoxLayout(size_hint_y=None, height=48)
        self.back = Button(text="< Nazad", size_hint_x=0.3)
        self.back.bind(on_press=lambda i: setattr(self.manager, "current", "main"))
        self.title_lbl = Label(text="Otchet", size_hint_x=0.7)
        top.add_widget(self.back)
        top.add_widget(self.title_lbl)
        tabs = BoxLayout(size_hint_y=None, height=44)
        for key, txt in [("ai", "AI"), ("deep", "Deep"), ("council", "Sovet"), ("verdict", "Verdikt")]:
            b = Button(text=txt)
            b.bind(on_press=lambda i, k=key: self.load(k))
            tabs.add_widget(b)
        self.scroll = ScrollView()
        self.label = Label(size_hint_y=None, halign="left", valign="top",
                           text_size=(Window.width - 20, None))
        self.label.bind(texture_size=self.label.setter("size"))
        self.scroll.add_widget(self.label)
        root.add_widget(top)
        root.add_widget(tabs)
        root.add_widget(self.scroll)
        self.add_widget(root)
        self.row = None
        self.args = None

    def open(self, row, args):
        self.row = row
        self.args = args
        self.title_lbl.text = row["symbol"]
        self.load("ai")

    def load(self, kind):
        self.label.text = "Zagruzka..."

        def work():
            text = ""
            try:
                if kind == "ai" and HAS_AGENTS:
                    text = ai_agents.build_report(self.row)
                elif kind == "deep" and HAS_DEEP:
                    text = "\n".join(deep_analysis.analyze_symbol_deep(
                        self.row["symbol"], self.args.interval, self.args.limit, self.args))
                elif kind == "council" and HAS_COUNCIL:
                    df = fast_scan.get_candles(self.row["symbol"], self.args.interval, self.args.limit)
                    if df is not None:
                        df = czs.add_indicators(df)
                        res = chart_ai.council(df.tail(220).reset_index(drop=True), self.row["symbol"])
                        text = chart_ai.report_text(self.row["symbol"], res)
                elif kind == "verdict" and HAS_BRAIN:
                    text = brain.verdict_report(self.row, self.args)
                else:
                    text = "Modul nedostupen."
            except Exception as e:
                text = f"Oshibka: {e}"
            Clock.schedule_once(lambda dt: setattr(self.label, "text", text), 0)

        threading.Thread(target=work, daemon=True).start()


class MainScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.df = None
        self.args = None
        root = BoxLayout(orientation="vertical")
        top = BoxLayout(size_hint_y=None, height=48)
        self.spin_interval = Spinner(text="4h", values=("1h", "4h", "1d"), size_hint_x=0.25)
        self.spin_top = Spinner(text="30", values=("10", "30", "50"), size_hint_x=0.25)
        self.btn_scan = Button(text="Skan")
        self.btn_scan.bind(on_press=self.start_scan)
        top.add_widget(self.spin_interval)
        top.add_widget(self.spin_top)
        top.add_widget(self.btn_scan)
        self.status = Label(text="Gotov", size_hint_y=None, height=30)
        self.scroll = ScrollView()
        self.grid = GridLayout(cols=1, size_hint_y=None, spacing=4)
        self.grid.bind(minimum_height=self.grid.setter("height"))
        self.scroll.add_widget(self.grid)
        root.add_widget(top)
        root.add_widget(self.status)
        root.add_widget(self.scroll)
        self.add_widget(root)

    def start_scan(self, *a):
        self.status.text = "Skanirovanie..."
        self.btn_scan.disabled = True

        def work():
            try:
                args = make_args(self.spin_interval.text, int(self.spin_top.text))
                self.args = args
                df = fast_scan.run_scan(args)
                if df is not None and not df.empty:
                    if HAS_AGENTS:
                        df = ai_agents.analyze_dataframe(df)
                    if HAS_BRAIN:
                        df = brain.enrich_dataframe(df, args, cap=10)
                Clock.schedule_once(lambda dt: self.fill(df), 0)
            except Exception as e:
                Clock.schedule_once(lambda dt: setattr(self.status, "text", f"Oshibka: {e}"), 0)
            finally:
                Clock.schedule_once(lambda dt: setattr(self.btn_scan, "disabled", False), 0)

        threading.Thread(target=work, daemon=True).start()

    def fill(self, df):
        self.df = df
        self.grid.clear_widgets()
        if df is None or df.empty:
            self.status.text = "Net signalov"
            return
        self.status.text = f"Par: {len(df)}"
        for _, r in df.head(40).iterrows():
            prob = r.get("prob", "")
            txt = (f"{r['symbol']}  {r['signal']}  sila {r['score']:.1f}"
                   + (f"  prob {prob}%" if prob != "" else ""))
            b = Button(text=txt, size_hint_y=None, height=56, halign="left")
            b.bind(on_press=lambda i, row=r: self.open_detail(row))
            self.grid.add_widget(b)

    def open_detail(self, row):
        d = self.manager.get_screen("detail")
        self.manager.current = "detail"
        d.open(row.to_dict(), self.args)


class ScannerApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainScreen(name="main"))
        sm.add_widget(DetailScreen(name="detail"))
        return sm


ScannerApp().run()
