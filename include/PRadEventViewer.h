#ifndef PRAD_EVENT_VIEWER_H
#define PRAD_EVENT_VIEWER_H

#include <QMainWindow>
#include <QFileDialog>
#include <QFutureWatcher>
#include <vector>

class HyCalScene;
class HyCalView;
class HyCalModule;
class Spectrum;
class SpectrumSettingPanel;
class PRadETChannel;
class PRadHistCanvas;
class PRadDataHandler;
class ETSettingPanel;
class PRadLogBox;
class PRadHVSystem;

//class GEM;
class TCanvas;
class TObject;
class TH1D;

QT_BEGIN_NAMESPACE
class QPushButton;
class QComboBox;
class QSpinBox;
class QSlider;
class QString;
class QLabel;
class QSplitter;
class QTreeWidget;
class QTreeWidgetItem;
class QTimer;
class QAction;
QT_END_NAMESPACE

enum HistType {
    ModuleHist,
    LMSHist,
};

enum AnnoType {
    NoAnnotation,
    ShowID,
    ShowDAQ,
    ShowTDC,
};

enum ViewMode {
    EnergyView,
    OccupancyView,
    PedestalView,
    SigmaView,
    HighVoltageView,
};

enum ViewerStatus {
    NO_INPUT,
    DATA_FILE,
    ONLINE_MODE,
};

class PRadEventViewer : public QMainWindow
{
    Q_OBJECT

public:
    PRadEventViewer();
    virtual ~PRadEventViewer();
    void ModuleAction(void (HyCalModule::*act)());
    void ListModules();
    ViewMode GetViewMode() {return viewMode;};
    AnnoType GetAnnoType() {return annoType;};
    QColor GetColor(const double &val);
    void UpdateStatusBar(ViewerStatus mode);
    void UpdateStatusInfo();
    void UpdateHistCanvas();
    void SelectModule(HyCalModule* module);

public slots:
    void Refresh();

private slots:
    void openFile();
    void openPedFile();
    void saveHistToFile();
    void takeSnapShot();
    void changeHistType(int index);
    void changeAnnoType(int index);
    void changeViewMode(int index);
    void changeSpectrumSetting();
    void changeCurrentEvent(int evt);
    void eraseBufferAction();
    void onlineUpdate();
    void initOnlineMode();
    bool connectETClient();
    void startOnlineMode();
    void stopOnlineMode();
    void connectHVSystem();
    void initHVSystem();
    void disconnectHVSystem();
    void startHVMonitor();
    void handleRootEvents();

private:
    void initView();
    void setupUI();
    void generateSpectrum();
    void generateHyCalModules();
    void buildModuleMap();
    void setupOnlineMode();
    void readModuleList();
    void readTDCList();
    void readSpecialChannels();
    void readPedestalData(const QString &filename);
    void eraseModuleBuffer();
    void createMainMenu();
    void createControlPanel();
    void createStatusBar();
    void createStatusWindow();
    void setupInfoWindow();
    void updateEventRange();
    void readEventFromFile(const QString &filepath);
    void fitEventsForPedestal();
    bool onlineSettings();
    QString getFileName(const QString &title,
                        const QString &dir,
                        const QStringList &filter,
                        const QString &suffix,
                        QFileDialog::AcceptMode mode = QFileDialog::AcceptOpen);

    PRadDataHandler *handler;
    int currentEvent;
    HistType histType;
    AnnoType annoType;
    ViewMode viewMode;

    HyCalModule *selection;
    Spectrum *energySpectrum;
    //GEM *myGEM;
    PRadETChannel *etChannel;
    PRadHVSystem *hvSystem;
    HyCalScene *HyCal;
    HyCalView *view;
    PRadHistCanvas *histCanvas;

    QString fileName;
    QTimer *onlineTimer;

    QSplitter *statusWindow;
    QSplitter *rightPanel;
    QSplitter *mainSplitter;

    QWidget *controlPanel;
    QSpinBox *eventSpin;
    QLabel *eventCntLabel;
    QComboBox *histTypeBox;
    QComboBox *annoTypeBox;
    QComboBox *viewModeBox;
    QPushButton *spectrumSettingButton;

    QTreeWidget *statusInfoWidget;
    QTreeWidgetItem *statusItem[5];

    QLabel *lStatusLabel;
    QLabel *rStatusLabel;

    QAction *openDataAction;
    QAction *openPedAction;
    QAction *onlineEnAction;
    QAction *onlineDisAction;
    QAction *hvEnableAction;
    QAction *hvMonitorAction;
    QAction *hvDisableAction;

    QFileDialog *fileDialog;
    ETSettingPanel *etSetting;
    SpectrumSettingPanel *specSetting;
    PRadLogBox *logBox;

    QFuture<bool> future;
    QFutureWatcher<void> watcher;
};

#endif
