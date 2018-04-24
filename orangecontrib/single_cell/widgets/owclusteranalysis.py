import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QStandardItemModel
from AnyQt.QtWidgets import QGridLayout
from Orange.data import (DiscreteVariable, Table, Domain)
from Orange.data.filter import Values, FilterDiscrete
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME, create_annotated_table
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.sql import check_sql_input

from orangecontrib.single_cell.preprocess.clusteranalysis import ClusterAnalysis
from orangecontrib.single_cell.widgets.contingency_table import ContingencyTable


class OWClusterAnalysis(widget.OWWidget):
    name = "Cluster Analysis"
    description = "Perform cluster analysis."
    priority = 2011

    inputs = [("Data", Table, "set_data", widget.Default)]
    outputs = [("Selected Data", Table),
               (ANNOTATED_DATA_SIGNAL_NAME, Table),
               ("Contingency Table", Table)]

    settingsHandler = DomainContextHandler(metas_in_res=True)
    rows = ContextSetting(None)
    columns = ContextSetting(None)
    clustering_var = ContextSetting(None)
    selection = ContextSetting(set())
    gene_selection = ContextSetting(0)
    n_genes_per_cluster = ContextSetting(3)
    n_most_enriched = ContextSetting(20)
    auto_apply = Setting(True)

    want_main_area = True

    def __init__(self):
        super().__init__()

        self.data = None
        self.feature_model = DomainModel(valid_types=DiscreteVariable)
        self.table = None

        box = gui.vBox(self.controlArea, "Info")
        self.infobox = gui.widgetLabel(box, self._get_info_string(None))

        box = gui.vBox(self.controlArea, "Rows")
        gui.comboBox(box, self, "clustering_var", sendSelectedValue=True,
                     model=self.feature_model, callback=self._run_cluster_analysis)

        layout = QGridLayout()
        bg = gui.radioButtonsInBox(
            self.controlArea, self, "gene_selection", orientation=layout,
            box="Gene Selection", callback=self._run_cluster_analysis)

        layout.addWidget(
            gui.appendRadioButton(bg, "", addToLayout=False), 1, 1)
        cb = gui.hBox(None, margin=0)
        gui.widgetLabel(cb, "Top")
        gui.spin(
            cb, self, "n_genes_per_cluster", minv=2, maxv=30,
            controlWidth=60, alignment=Qt.AlignRight)
        gui.widgetLabel(cb, "genes per cluster")
        gui.rubber(cb)
        layout.addWidget(cb, 1, 2, Qt.AlignLeft)

        layout.addWidget(
            gui.appendRadioButton(bg, "", addToLayout=False), 2, 1)
        mb = gui.hBox(None, margin=0)
        gui.widgetLabel(mb, "Top")
        gui.spin(
            mb, self, "n_most_enriched", minv=2, maxv=30,
            controlWidth=60, alignment=Qt.AlignRight)
        gui.widgetLabel(mb, "highest enrichments")
        gui.rubber(mb)
        layout.addWidget(mb, 2, 2, Qt.AlignLeft)

        layout.addWidget(
            gui.appendRadioButton(bg, "", addToLayout=False), 3, 1)
        sb = gui.hBox(None, margin=0)
        gui.widgetLabel(sb, "User-provided list of genes")
        gui.rubber(sb)
        layout.addWidget(sb, 3, 2)

        gui.rubber(self.controlArea)

        self.apply_button = gui.auto_commit(
            self.controlArea, self, "auto_apply", "&Apply", box=False)

        self.tablemodel = QStandardItemModel(self)
        self.tableview = ContingencyTable(self, self.tablemodel)
        self.mainArea.layout().addWidget(self.tableview)

    def _get_info_string(self, cluster_variable):
        formatstr = "Cells: {0}\nGenes: {1}\nClusters: {2}"
        if self.data:
            return formatstr.format(len(self.data),
                                    len(self.data.domain.attributes),
                                    len(self.data.domain[cluster_variable].values))
        else:
            return formatstr.format(*["No input data"]*3)

    @check_sql_input
    def set_data(self, data):
        if self.feature_model:
            self.closeContext()
        self.data = data
        self.feature_model.set_domain(None)
        self.rows = None
        self.columns = None
        if self.data:
            self.feature_model.set_domain(self.data.domain)
            if self.feature_model:
                self.clustering_var = self.feature_model[0]
                self._run_cluster_analysis()
            else:
                self.tablemodel.clear()
        else:
            self.tablemodel.clear()

    def _run_cluster_analysis(self):
        self.infobox.setText(self._get_info_string(self.clustering_var.name))
        CA = ClusterAnalysis(self.data, self.clustering_var.name)
        if self.gene_selection == 0:
            CA.enriched_genes_per_cluster(self.n_genes_per_cluster)
        elif self.gene_selection == 1:
            CA.enriched_genes_data(self.n_most_enriched)
        elif self.gene_selection == 2:
            pass
        CA.percentage_expressing()
        self.table = CA.sort_percentage_expressing()
        # Referencing the variable in the table directly doesn't preserve the order of clusters.
        self.clusters = [self.clustering_var.values[ix] for ix in self.table.get_column_view(self.clustering_var.name)[0]]
        genes = [var.name for var in self.table.domain.variables]
        self.rows = self.clustering_var
        self.columns = DiscreteVariable("Gene", genes, ordered=True)
        self.tableview.set_headers(self.clusters, self.columns.values, circles=True)
        self.tableview.update_table(self.table.X, formatstr="{:.2f}")
        self._invalidate()

    def handleNewSignals(self):
        self._invalidate()

    def commit(self):
        if len(self.selection):
            cluster_ids = set()
            column_ids = set()
            for (ir, ic) in self.selection:
                cluster_ids.add(ir)
                column_ids.add(ic)
            new_domain = Domain([self.data.domain[self.columns.values[col]] for col in column_ids],
                                self.data.domain.class_vars,
                                self.data.domain.metas)
            selected_data = Values([FilterDiscrete(self.clustering_var, [self.clustering_var.values[ir]])
                                    for ir in cluster_ids],
                                   conjunction=False)(self.data)
            selected_data = selected_data.transform(new_domain)
            annotated_data = create_annotated_table(self.data,
                                                    np.where(np.in1d(self.data.ids, selected_data.ids, True)))
        else:
            selected_data = None
            annotated_data = create_annotated_table(self.data, [])
        self.send("Selected Data", selected_data)
        self.send(ANNOTATED_DATA_SIGNAL_NAME, annotated_data)
        self.send("Contingency Table", self.table)

    def _invalidate(self):
        self.selection = self.tableview.get_selection()
        self.commit()

    def send_report(self):
        rows = None
        columns = None
        if self.data is not None:
            rows = self.rows
            if rows in self.data.domain:
                rows = self.data.domain[rows]
            columns = self.columns
            if columns in self.data.domain:
                columns = self.data.domain[columns]
        self.report_items((
            ("Rows", rows),
            ("Columns", columns),
        ))


def test():
    from AnyQt.QtWidgets import QApplication
    app = QApplication([])

    w = OWClusterAnalysis()
    data = Table("../../../../testdata.tab")
    data.X = data.X > 0
    w.set_data(data)
    w.handleNewSignals()
    w.show()
    app.exec_()


if __name__ == "__main__":
    test()
