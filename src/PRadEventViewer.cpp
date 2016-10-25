//============================================================================//
// Main class for PRad Event Viewer, derived from QMainWindow                 //
//                                                                            //
// Chao Peng, Weizhi Xiong                                                    //
// 02/27/2016                                                                 //
//============================================================================//

#include "PRadEventViewer.h"

#include "TApplication.h"
#include "TSystem.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TSpectrum.h"

#include <utility>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unordered_map>

#if QT_VERSION >= 0x050000
#include <QtWidgets>
#include <QtConcurrent>
#else
#include <QtGui>
#endif

#include "HyCalModule.h"
#include "HyCalScene.h"
#include "HyCalView.h"
#include "Spectrum.h"
#include "SpectrumSettingPanel.h"
#include "HtmlDelegate.h"
#include "ConfigParser.h"

#include "PRadEvioParser.h"
#include "PRadHistCanvas.h"
#include "PRadDataHandler.h"
#include "PRadDAQUnit.h"
#include "PRadTDCGroup.h"
#include "PRadLogBox.h"
#include "PRadBenchMark.h"

#ifdef RECON_DISPLAY
#include "PRadHyCalCluster.h"
#include "PRadIslandCluster.h"
#include "PRadSquareCluster.h"
#include "PRadCoordSystem.h"
#include "PRadDetMatch.h"
#include "PRadGEMSystem.h"
#include "ReconSettingPanel.h"
#endif

#ifdef USE_ONLINE_MODE
#include "PRadETChannel.h"
#include "ETSettingPanel.h"
#endif

#ifdef USE_CAEN_HV
#include "PRadHVSystem.h"
#endif

#ifdef USE_EVIO_LIB
#include "evioUtil.hxx"
#include "evioFileChannel.hxx"
#endif

#define cap_value(a, min, max) \
        (((a) >= (max)) ? (max) : ((a) <= (min)) ? (min) : (a))

//============================================================================//
// constructor                                                                //
//============================================================================//
PRadEventViewer::PRadEventViewer()
: handler(new PRadDataHandler())
{
    initView();
    setupUI();
}

PRadEventViewer::~PRadEventViewer()
{
#ifdef USE_ONLINE_MODE
    delete etChannel;
#endif
#ifdef USE_CAEN_HV
    delete hvSystem;
#endif
#ifdef RECON_DISPLAY
    delete coordSystem;
    delete detMatch;
#endif
    delete handler;
}

// set up the view for HyCal
void PRadEventViewer::initView()
{
    HyCal = new HyCalScene(this, -800, -800, 1600, 1600);
    HyCal->setBackgroundBrush(QColor(255, 255, 238));

    generateScalerBoxes();
    generateSpectrum();
    generateHyCalModules();

    view = new HyCalView;
    view->setScene(HyCal);

    // root timer to process root events
    QTimer *rootTimer = new QTimer(this);
    connect(rootTimer, SIGNAL(timeout()), this, SLOT(handleRootEvents()));
    rootTimer->start(50);

    // setup optional components
#ifdef RECON_DISPLAY
    setupReconDisplay();
#endif

#ifdef USE_ONLINE_MODE
    setupOnlineMode();
#endif

#ifdef USE_CAEN_HV
    setupHVSystem();
#endif
}

// set up the UI
void PRadEventViewer::setupUI()
{
    setWindowTitle(tr("PRad Event Viewer"));
    QDesktopWidget dw;
    double height = dw.height();
    double width = dw.width();
    double scale = (width/height > (16./9.))? 0.8 : 0.8 * ((width/height)/(16/9.));
    view->scale((height*scale)/1440, (height*scale)/1440);
    resize(height*scale*16./9., height*scale);

    createMainMenu();
    createStatusBar();

    createControlPanel();
    createStatusWindow();

    rightPanel = new QSplitter(Qt::Vertical);
    rightPanel->addWidget(statusWindow);
    rightPanel->addWidget(controlPanel);
    rightPanel->setStretchFactor(0,7);
    rightPanel->setStretchFactor(1,2);

    mainSplitter = new QSplitter(Qt::Horizontal);
    mainSplitter->addWidget(view);
    mainSplitter->addWidget(rightPanel);
    mainSplitter->setStretchFactor(0,2);
    mainSplitter->setStretchFactor(1,3);

    setCentralWidget(mainSplitter);

    fileDialog = new QFileDialog();
}

//============================================================================//
// generate elements                                                          //
//============================================================================//

// create spectrum
void PRadEventViewer::generateSpectrum()
{
    energySpectrum = new Spectrum(40, 1100, 1, 1000);
    energySpectrum->setPos(600, 0);
    HyCal->addItem(energySpectrum);

    specSetting = new SpectrumSettingPanel(this);
    specSetting->ConnectSpectrum(energySpectrum);

    connect(energySpectrum, SIGNAL(spectrumChanged()), this, SLOT(Refresh()));
}

// crate HyCal modules from module list
void PRadEventViewer::generateHyCalModules()
{
    readModuleList();

    // other information for data handler
    handler->ReadEPICSChannels("config/epics_channels.txt");
    handler->ReadPedestalFile("config/pedestal.dat");
    handler->ReadCalibrationFile("config/calibration.txt");

    // Default setting
    selection = HyCal->GetModuleList().at(0);
    annoType = NoAnnotation;
    viewMode = EnergyView;

}

void PRadEventViewer::generateScalerBoxes()
{
    HyCal->AddScalerBox(tr("Pb-Glass Sum")    , Qt::black, QRectF(-650, -640, 150, 40), QColor(255, 155, 155, 50));
    HyCal->AddScalerBox(tr("Total Sum")       , Qt::black, QRectF(-500, -640, 150, 40), QColor(155, 255, 155, 50));
    HyCal->AddScalerBox(tr("LMS Led")         , Qt::black, QRectF(-350, -640, 150, 40), QColor(155, 155, 255, 50));
    HyCal->AddScalerBox(tr("LMS Alpha")       , Qt::black, QRectF(-200, -640, 150, 40), QColor(255, 200, 100, 50));
    HyCal->AddScalerBox(tr("Master Or")       , Qt::black, QRectF( -50, -640, 150, 40), QColor(100, 255, 200, 50));
    HyCal->AddScalerBox(tr("Scintillator")    , Qt::black, QRectF( 100, -640, 150, 40), QColor(200, 100, 255, 50));
    HyCal->AddScalerBox(tr("Live Time")       , Qt::black, QRectF( 250, -640, 150, 40), QColor(200, 255, 100, 50));
    HyCal->AddScalerBox(tr("Beam Current")    , Qt::black, QRectF( 400, -640, 150, 40), QColor(100, 200, 255, 50));
}

//============================================================================//
// create menu and tool box                                                   //
//============================================================================//

// main menu
void PRadEventViewer::createMainMenu()
{
    menuBar()->addMenu(setupFileMenu());

    menuBar()->addMenu(setupCalibMenu());

    menuBar()->addMenu(setupToolMenu());

    // menu for optional components
#ifdef RECON_DISPLAY
    menuBar()->addMenu(setupReconMenu());
#endif

#ifdef USE_ONLINE_MODE
    menuBar()->addMenu(setupOnlineMenu());
#endif

#ifdef USE_CAEN_HV
    menuBar()->addMenu(setupHVMenu());
#endif

}

// file menu, open, save, quit
QMenu *PRadEventViewer::setupFileMenu()
{
    QMenu *fileMenu = new QMenu(tr("&File"));

    openDataAction = fileMenu->addAction(tr("&Open Data File"));
    openDataAction->setShortcuts(QKeySequence::Open);

    QAction *openPedAction = fileMenu->addAction(tr("Open &Pedestal File"));
    openPedAction->setShortcuts(QKeySequence::Print);

    QAction *saveHistAction = fileMenu->addAction(tr("Save &Histograms"));
    saveHistAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_H));

    QAction *savePedAction = fileMenu->addAction(tr("&Save Pedestal File"));
    savePedAction->setShortcuts(QKeySequence::Save);

    QAction *quitAction = fileMenu->addAction(tr("&Quit"));
    quitAction->setShortcuts(QKeySequence::Quit);

    connect(openDataAction, SIGNAL(triggered()), this, SLOT(openDataFile()));
    connect(openPedAction, SIGNAL(triggered()), this, SLOT(openPedFile()));
    connect(saveHistAction, SIGNAL(triggered()), this, SLOT(saveHistToFile()));
    connect(savePedAction, SIGNAL(triggered()), this, SLOT(savePedestalFile()));
    connect(quitAction, SIGNAL(triggered()), qApp, SLOT(quit()));

    return fileMenu;
}


// calibration related menu
QMenu *PRadEventViewer::setupCalibMenu()
{
    QMenu *caliMenu = new QMenu(tr("&Calibration"));

    QAction *initializeAction = caliMenu->addAction(tr("Initialize From Data File"));

    QAction *openCalFileAction = caliMenu->addAction(tr("Read Calibration Constants"));

    QAction *openGainFileAction = caliMenu->addAction(tr("Normalize Gain From File"));

    QAction *correctGainAction = caliMenu->addAction(tr("Normalize Gain From Data"));

    QAction *fitPedAction = caliMenu->addAction(tr("Update Pedestal From Data"));

    connect(initializeAction, SIGNAL(triggered()), this, SLOT(initializeFromFile()));
    connect(openCalFileAction, SIGNAL(triggered()), this, SLOT(openCalibrationFile()));
    connect(openGainFileAction, SIGNAL(triggered()), this, SLOT(openGainFactorFile()));
    connect(correctGainAction, SIGNAL(triggered()), this, SLOT(correctGainFactor()));
    connect(fitPedAction, SIGNAL(triggered()), this, SLOT(fitPedestal()));

    return caliMenu;
}

// tool menu, useful tools
QMenu *PRadEventViewer::setupToolMenu()
{
    QMenu *toolMenu = new QMenu(tr("&Tools"));

    QAction *eraseAction = toolMenu->addAction(tr("Erase Buffer"));
    eraseAction->setShortcut(QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_X));

    QAction *findPeakAction = toolMenu->addAction(tr("Find Peak"));
    findPeakAction->setShortcut(QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_F));

    QAction *fitHistAction = toolMenu->addAction(tr("Fit Histogram"));
    fitHistAction->setShortcut(QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_H));

    QAction *snapShotAction = toolMenu->addAction(tr("Take SnapShot"));
    snapShotAction->setShortcut(QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_S));

    QAction *showCustomAction = toolMenu->addAction(tr("Show Custom Map"));
    showCustomAction->setShortcut(QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_C));

    QAction *findEventAction = toolMenu->addAction(tr("Find Event"));
    findEventAction->setShortcut(QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_E));

    connect(eraseAction, SIGNAL(triggered()), this, SLOT(eraseBufferAction()));
    connect(findPeakAction, SIGNAL(triggered()), this, SLOT(findPeak()));
    connect(fitHistAction, SIGNAL(triggered()), this, SLOT(fitHistogram()));
    connect(snapShotAction, SIGNAL(triggered()), this, SLOT(takeSnapShot()));
    connect(showCustomAction, SIGNAL(triggered()), this, SLOT(openCustomMap()));
    connect(findEventAction, SIGNAL(triggered()), this, SLOT(findEvent()));

    return toolMenu;
}

// tool box
void PRadEventViewer::createControlPanel()
{
    eventSpin = new QSpinBox;
    eventSpin->setRange(0, 0);
    eventSpin->setPrefix("Event # ");
    connect(eventSpin, SIGNAL(valueChanged(int)),
            this, SLOT(changeCurrentEvent(int)));
    connect(this, SIGNAL(currentEventChanged(int)),
            this, SLOT(handleEventChange(int)));

    histTypeBox = new QComboBox();
    histTypeBox->addItem(tr("Energy&TDC Hist"));
    histTypeBox->addItem(tr("Module Hist"));
    histTypeBox->addItem(tr("Tagger Hist"));
    annoTypeBox = new QComboBox();
    annoTypeBox->addItem(tr("No Annotation"));
    annoTypeBox->addItem(tr("Module ID"));
    annoTypeBox->addItem(tr("DAQ Info"));
    annoTypeBox->addItem(tr("Show TDC Group"));
    viewModeBox = new QComboBox();
    viewModeBox->addItem(tr("Energy View"));
    viewModeBox->addItem(tr("Occupancy View"));
    viewModeBox->addItem(tr("Pedestal View"));
    viewModeBox->addItem(tr("Ped. Sigma View"));
    viewModeBox->addItem(tr("High Voltage View"));
    viewModeBox->addItem(tr("HV Setting View"));
    viewModeBox->addItem(tr("Custom Map View"));

    spectrumSettingButton = new QPushButton("Spectrum Settings");

    eventCntLabel = new QLabel;
    eventCntLabel->setText(tr("No events data loaded."));

    connect(histTypeBox, SIGNAL(currentIndexChanged(int)),
            this, SLOT(changeHistType(int)));
    connect(annoTypeBox, SIGNAL(currentIndexChanged(int)),
            this, SLOT(changeAnnoType(int)));
    connect(viewModeBox, SIGNAL(currentIndexChanged(int)),
            this, SLOT(changeViewMode(int)));
    connect(spectrumSettingButton, SIGNAL(clicked()),
            this, SLOT(changeSpectrumSetting()));

    logBox = new PRadLogBox();

    QGridLayout *layout = new QGridLayout();

    layout->addWidget(eventSpin,             0, 0, 1, 1);
    layout->addWidget(eventCntLabel,         0, 1, 1, 1);
    layout->addWidget(spectrumSettingButton, 0, 2, 1, 1);
    layout->addWidget(histTypeBox,           1, 0, 1, 1);
    layout->addWidget(viewModeBox,           1, 1, 1, 1);
    layout->addWidget(annoTypeBox,           1, 2, 1, 1);
    layout->addWidget(logBox,                2, 0, 3, 3);

    controlPanel = new QWidget(this);
    controlPanel->setLayout(layout);

}

// status bar
void PRadEventViewer::createStatusBar()
{
    lStatusLabel = new QLabel(tr("Please open a data file or use online mode."));
    lStatusLabel->setAlignment(Qt::AlignLeft);
    lStatusLabel->setMinimumSize(lStatusLabel->sizeHint());

    rStatusLabel = new QLabel(tr(""));
    rStatusLabel->setAlignment(Qt::AlignRight);


    statusBar()->addPermanentWidget(lStatusLabel, 1);
    statusBar()->addPermanentWidget(rStatusLabel, 1);
}

// Status window
void PRadEventViewer::createStatusWindow()
{
    statusWindow = new QSplitter(Qt::Vertical);

    // status info part
    setupInfoWindow();
    histCanvas = new PRadHistCanvas(this);
    histCanvas->AddCanvas(0, 0, 38);
    histCanvas->AddCanvas(1, 0, 46);
    histCanvas->AddCanvas(2, 0, 30);

    statusWindow->addWidget(statusInfoWidget);
    statusWindow->addWidget(histCanvas);
}

// status infor window
void PRadEventViewer::setupInfoWindow()
{
    statusInfoWidget = new QTreeWidget;
    statusInfoWidget->setSelectionMode(QAbstractItemView::NoSelection);
    QStringList statusInfoTitle;
    QFont font("arial", 10 , QFont::Bold );

    statusInfoTitle << tr("  Module Property") << tr("  Value  ")
                    << tr("  Module Property") << tr("  Value  ");
    statusInfoWidget->setHeaderLabels(statusInfoTitle);
    statusInfoWidget->setItemDelegate(new HtmlDelegate());
    statusInfoWidget->setIndentation(0);
    statusInfoWidget->setMaximumHeight(180);

    // add new items to status info
    QStringList statusProperty;
    statusProperty << tr("  Module ID") << tr("  Module Type") << tr("  DAQ Address") << tr("  TDC Group") << tr("  HV Address") << tr("  Occupancy")
                   << tr("  Pedestal") << tr("  Event Number") << tr("  Energy") << tr("  ADC Count") << tr("  High Voltage") << tr("  Custom (Editable)");

    for(int i = 0; i < 6; ++i) // row iteration
    {
        statusItem[i] = new QTreeWidgetItem(statusInfoWidget);
        for(int j = 0; j < 4; ++ j) // column iteration
        {
            if(j&1) { // odd column
                statusItem[i]->setFont(j, font);
            } else { // even column
                statusItem[i]->setText(j, statusProperty.at(6*j/2 + i));
            }
            if(i&1) { // even row
                statusItem[i]->setBackgroundColor(j, QColor(255,255,208));
            }
        }
    }

    // Spectial rule to enable html text support for subscript
    statusItem[1]->useHtmlDelegate(1);

    // set the custom value editable
    connect(statusInfoWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*,int)), this, SLOT(editCustomValueLabel(QTreeWidgetItem*,int)));

    statusInfoWidget->resizeColumnToContents(0);
    statusInfoWidget->resizeColumnToContents(2);

}

//============================================================================//
// read information from configuration files                                  //
//============================================================================//

// read module list from file
void PRadEventViewer::readModuleList()
{
    // build TDC groups first
    handler->ReadTDCList("config/tdc_group_list.txt");

    ConfigParser c_parser;
    if(!c_parser.OpenFile("config/module_list.txt")) {
        std::cerr << "ERROR: Missing configuration file \"config/module_list.txt\""
                  << ", cannot generate HyCal channels!"
                  << std::endl;
        exit(1);
    }

    std::string moduleName;
    std::string tdcGroup;
    unsigned int crate, slot, channel, type;
    double size_x, size_y, x, y;

    // some info that is not read from list
    // initialize first

    while (c_parser.ParseLine())
    {
        if(c_parser.NbofElements() == 13) {
            c_parser >> moduleName // module name
                     >> crate >> slot >> channel // daq settings
                     >> tdcGroup // tdc group name
                     >> type >> size_x >> size_y >> x >> y; // geometry

            ChannelAddress daqAddr(crate, slot, channel);
            PRadDAQUnit::Geometry geo(PRadDAQUnit::ChannelType(type), size_x, size_y, x, y);

            HyCalModule* newModule = new HyCalModule(this, moduleName, daqAddr, tdcGroup, geo);

            c_parser >> crate >> slot >> channel; // hv settings
            ChannelAddress hvAddr(crate, slot, channel);
            newModule->UpdateHVSetup(hvAddr);

            HyCal->addModule(newModule);
            handler->RegisterChannel(newModule);
        } else {
            std::cout << "Unrecognized input format in configuration file, skipped one line!"
                      << std::endl;
        }
    }

    c_parser.CloseFile();

    // make handler to build the module map
    handler->BuildChannelMap();

    // set TDC Group box for the TDC view
    setTDCGroupBox();
}

// build module maps for speed access to module
// send the tdc group geometry to scene for annotation
void PRadEventViewer::setTDCGroupBox()
{
    // tdc maps
    std::unordered_map< std::string, PRadTDCGroup * > tdcList = handler->GetTDCGroupSet();
    for(auto &it : tdcList)
    {
        PRadTDCGroup *tdcGroup = it.second;
        std::vector< PRadDAQUnit* > groupList = tdcGroup->GetGroupList();

        if(!groupList.size())
            continue;

        // get id and set background color
        QString tdcGroupName = QString::fromStdString(tdcGroup->GetName());
        QColor bkgColor;
        int tdc = tdcGroupName.mid(1).toInt();
        if(tdcGroupName.at(0) == 'G') { // below is to make different color for adjacent groups
             if(tdc&1)
                bkgColor = QColor(255, 153, 153, 50);
             else
                bkgColor = QColor(204, 204, 255, 50);
        } else {
            if((tdc&1)^(((tdc-1)/6)&1))
                bkgColor = QColor(51, 204, 255, 50);
            else
                bkgColor = QColor(0, 255, 153, 50);
        }

        // get the tdc group box size
        double xmax = -600., xmin = 600.;
        double ymax = -600., ymin = 600.;
        bool has_module = false;
        for(auto &channel : groupList)
        {
            HyCalModule *module = dynamic_cast<HyCalModule *>(channel);
            if(module == nullptr)
                continue;

            has_module = true;
            PRadDAQUnit::Geometry geo = module->GetGeometry();
            xmax = std::max(geo.x + geo.size_x/2., xmax);
            xmin = std::min(geo.x - geo.size_x/2., xmin);
            ymax = std::max(geo.y + geo.size_y/2., ymax);
            ymin = std::min(geo.y - geo.size_y/2., ymin);
        }
        QRectF groupBox = QRectF(xmin + HYCAL_SHIFT, ymin, xmax-xmin, ymax-ymin);
        if(has_module)
            HyCal->AddTDCBox(tdcGroupName, Qt::black, groupBox, bkgColor);
    }
}

//============================================================================//
// HyCal Modules Actions                                                      //
//============================================================================//

// do the action for all modules
template<typename... Args>
void PRadEventViewer::ModuleAction(void (HyCalModule::*act)(Args...), Args&&... args)
{
    QVector<HyCalModule*> moduleList = HyCal->GetModuleList();
    for(auto &module : moduleList)
    {
        (module->*act)(std::forward<Args>(args)...);
    }
}

void PRadEventViewer::ListModules()
{
    QVector<HyCalModule*> moduleList = HyCal->GetModuleList();
    std::ofstream outf("config/current_list.txt");
    outf << "#" << std::setw(9) << "Name"
         << std::setw(10) << "DAQ Crate"
         << std::setw(6) << "Slot"
         << std::setw(6) << "Chan"
         << std::setw(6) << "TDC"
         << std::setw(10) << "Type"
         << std::setw(10) << "size_x"
         << std::setw(10) << "size_y"
         << std::setw(10) << "x"
         << std::setw(10) << "y"
         << std::setw(10) << "HV Crate"
         << std::setw(6) << "Slot"
         << std::setw(6) << "Chan"
         << std::endl;


    for(auto &module : moduleList)
    {
        outf << std::setw(10) << module->GetReadID().toStdString()
             << std::setw(10) << module->GetDAQInfo().crate
             << std::setw(6)  << module->GetDAQInfo().slot
             << std::setw(6)  << module->GetDAQInfo().channel
             << std::setw(6)  << module->GetTDCName()
             << std::setw(10) << (int)module->GetGeometry().type
             << std::setw(10)  << module->GetGeometry().size_x
             << std::setw(10)  << module->GetGeometry().size_y
             << std::setw(10)  << module->GetGeometry().x
             << std::setw(10)  << -module->GetGeometry().y
//             << std::setw(10) << module->GetPedestal().mean
//             << std::setw(8)  << module->GetPedestal().sigma
             << std::setw(10) << module->GetHVInfo().crate
             << std::setw(6)  << module->GetHVInfo().slot
             << std::setw(6)  << module->GetHVInfo().channel
             << std::endl;
    }
}

//============================================================================//
// Get color, refresh and erase                                               //
//============================================================================//

// get color from spectrum
QColor PRadEventViewer::GetColor(const double &val)
{
    return energySpectrum->GetColor(val);
}

// refresh all the view
void PRadEventViewer::Refresh()
{
    switch(viewMode)
    {
    case PedestalView:
        ModuleAction(&HyCalModule::ShowPedestal);
        break;
    case SigmaView:
        ModuleAction(&HyCalModule::ShowPedSigma);
        break;
    case OccupancyView:
        ModuleAction(&HyCalModule::ShowOccupancy);
        break;
#ifdef USE_CAEN_HV
    case HighVoltageView:
    {
        auto moduleList = HyCal->GetModuleList();
        for(auto module : moduleList)
        {
            ChannelAddress hv_addr = module->GetHVInfo();
            PRadHVSystem::Voltage volt = hvSystem->GetVoltage(hv_addr.crate, hv_addr.slot, hv_addr.channel);
            if(!volt.ON)
                module->SetColor(QColor(255, 255, 255));
            else
                module->SetColor(energySpectrum->GetColor(volt.Vmon));
        }
        break;
    }
    case VoltageSetView:
    {
        auto moduleList = HyCal->GetModuleList();
        for(auto module : moduleList)
        {
            ChannelAddress hv_addr = module->GetHVInfo();
            PRadHVSystem::Voltage volt = hvSystem->GetVoltage(hv_addr.crate, hv_addr.slot, hv_addr.channel);
            module->SetColor(energySpectrum->GetColor(volt.Vset));
        }
        break;
    }
#endif
    case EnergyView:
        ModuleAction(&HyCalModule::ShowEnergy);
        break;
    case CustomView:
        ModuleAction(&HyCalModule::ShowCustomValue);
        break;
    }

    UpdateStatusInfo();

    QWidget *viewport = view->viewport();
    viewport->update();
}

// clean all the data buffer
void PRadEventViewer::eraseData()
{
    handler->Clear();
    updateEventRange();
}

//============================================================================//
// functions that react to menu, tool                                         //
//============================================================================//

// open file
void PRadEventViewer::openDataFile()
{
    QString codaData;
    codaData.sprintf("%s", getenv("CODA_DATA"));
    if (codaData.isEmpty())
        codaData = QDir::currentPath();

    QStringList filters;
    filters << "Data files (*.dst *.ev *.evio *.evio.*)"
            << "All files (*)";

    QStringList fileList = getFileNames(tr("Choose a data file"), codaData, filters, "");

    if (fileList.isEmpty())
        return;

    eraseData();

    PRadBenchMark timer;

    for(QString &file : fileList)
    {
        //TODO, dialog to notice waiting
//        QtConcurrent::run(this, &PRadEventViewer::readEventFromFile, fileName);
        fileName = file;
        if(fileName.contains(".dst")) {
            handler->ReadFromDST(fileName.toStdString());
        } else {
            readEventFromFile(fileName);
        }
        UpdateStatusBar(DATA_FILE);
    }

    std::cout << "Parsed " << handler->GetEventCount() << " events and "
              << handler->GetEPICSEventCount() << " EPICS events from "
              << fileList.size() << " files." << std::endl
              << " Used " << timer.GetElapsedTime() << " ms."
              << std::endl;

    updateEventRange();
}

// open pedestal file
void PRadEventViewer::openPedFile()
{
    QString dir = QDir::currentPath() + "/config";

    QStringList filters;
    filters << "Data files (*.dat *.txt)"
            << "All files (*)";

    QString file = getFileName(tr("Open pedestal file"), dir, filters, "");

    if (!file.isEmpty()) {
        handler->ReadPedestalFile(file.toStdString());
    }
}

// initialize handler from data file
void PRadEventViewer::initializeFromFile()
{
    QString codaData;
    codaData.sprintf("%s", getenv("CODA_DATA"));
    if (codaData.isEmpty())
        codaData = QDir::currentPath();

    QStringList filters;
    filters << "Data files (*.dat *.ev *.evio *.evio.*)"
            << "All files (*)";

    QString file = getFileName(tr("Choose the first data file in a run"), codaData, filters, "");

    if (file.isEmpty())
        return;

    PRadBenchMark timer;

    handler->InitializeByData(file.toStdString());

    updateEventRange();

    std::cout << "Initialized data handler from file "
              << "\"" << file.toStdString() << "\"." << std::endl
              << " Used " << timer.GetElapsedTime() << " ms."
              << std::endl;
}

// open calibration factor file
void PRadEventViewer::openCalibrationFile()
{
    QString dir = QDir::currentPath() + "/config";

    QStringList filters;
    filters << "Data files (*.dat *.txt)"
            << "All files (*)";

    QString file = getFileName(tr("Open calibration constants file"), dir, filters, "");

    if (!file.isEmpty()) {
        handler->ReadCalibrationFile(file.toStdString());
    }
}

void PRadEventViewer::openGainFactorFile()
{
    QString dir = QDir::currentPath() + "/config";

    QStringList filters;
    filters << "Data files (*.dat *.txt)"
            << "All files (*)";

    QString file = getFileName(tr("Open gain factors file"), dir, filters, "");

    if (!file.isEmpty()) {
        handler->ReadGainFactor(file.toStdString());
    }
}

void PRadEventViewer::openCustomMap()
{
    QString dir = QDir::currentPath();

    QStringList filters;
    filters << "Data files (*.dat *.txt)"
            << "All files (*)";

    QString file = getFileName(tr("Open custom value file"), dir, filters, "");

    if (!file.isEmpty()) {
        readCustomValue(file);
    }
}

void PRadEventViewer::findEvent()
{
    QDialog dialog(this);
    // Use a layout allowing to have a label next to each field
    QFormLayout form(&dialog);

    // Add some text above the fields
    form.addRow(new QLabel("Find event from data bank:"));

    // Add the lineEdits with their respective labels
    QVector<QLineEdit *> fields;
    QString label = "Event Number: ";

    QLineEdit *lineEdit = new QLineEdit(&dialog);
    form.addRow(label, lineEdit);

    // Add some standard buttons (Cancel/Ok) at the bottom of the dialog
    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                               Qt::Horizontal, &dialog);
    form.addRow(&buttonBox);
    QObject::connect(&buttonBox, SIGNAL(accepted()), &dialog, SLOT(accept()));
    QObject::connect(&buttonBox, SIGNAL(rejected()), &dialog, SLOT(reject()));

    // Show the dialog as modal
    if (dialog.exec() == QDialog::Accepted) {
        // If the user didn't dismiss the dialog, do something with the fields
        int index = handler->FindEventIndex(lineEdit->text().toInt());
        if(index >= 0)
            eventSpin->setValue(index + 1);
        else {
            QMessageBox::critical(this, "Find Event", "Event " + lineEdit->text() + " is not found in bank.");
        }
    }

}

void PRadEventViewer::changeHistType(int index)
{
    histType = (HistType)index;
    UpdateHistCanvas();
}

void PRadEventViewer::changeAnnoType(int index)
{
    annoType = (AnnoType)index;
    Refresh();
}

void PRadEventViewer::changeViewMode(int index)
{
    viewMode = (ViewMode)index;
    specSetting->ChoosePreSetting(index);
    Refresh();
}

void PRadEventViewer::changeSpectrumSetting()
{
    if(specSetting->isVisible())
        specSetting->close();
    else
        specSetting->show();
}

void PRadEventViewer::eraseBufferAction()
{
    QMessageBox::StandardButton confirm;
    confirm = QMessageBox::question(this,
                                   "Erase Event Buffer",
                                   "Clear all the events, including histograms?",
                                    QMessageBox::Yes|QMessageBox::No);
    if(confirm == QMessageBox::Yes)
        eraseData();
}

void PRadEventViewer::UpdateStatusBar(ViewerStatus mode)
{
    QString statusText;
    switch(mode)
    {
    case NO_INPUT:
        statusText = tr("Please open a data file or use online mode.");
        break;
    case DATA_FILE:
        statusText = tr("Current Data File: ")+fileName;
        break;
    case ONLINE_MODE:
        statusText = tr("In online mode");
        break;
    }
    lStatusLabel->setText(statusText);
}

void PRadEventViewer::changeCurrentEvent(int evt)
{
    emit currentEventChanged(evt);
}

void PRadEventViewer::handleEventChange(int evt)
{
    handler->ChooseEvent(evt - 1); // fetch data from handler
#ifdef RECON_DISPLAY
    if(enableRecon->isChecked())
        showReconEvent(evt - 1);
#endif
    Refresh();
}

void PRadEventViewer::updateEventRange()
{
    int total = handler->GetEventCount();

    if(total) {
        eventCntLabel->setText(tr("Total events: ") + QString::number(total));
        eventSpin->setRange(1, total);
    } else {
        eventCntLabel->setText(tr("No events data loaded."));
        eventSpin->setRange(0, 0);
    }
    UpdateHistCanvas();

    emit currentEventChanged(eventSpin->value());
}

void PRadEventViewer::UpdateHistCanvas()
{
    gSystem->ProcessEvents();
    switch(histType) {
    default:
    case EnergyTDCHist:
        if(selection != nullptr) {
            histCanvas->UpdateHist(1, selection->GetHist("PHYS"));
            PRadTDCGroup *tdc = selection->GetTDCGroup();
            if(tdc)
                histCanvas->UpdateHist(2, tdc->GetHist());
            else
                histCanvas->UpdateHist(2, selection->GetHist("LMS"));
        }
        histCanvas->UpdateHist(3, handler->GetEnergyHist());
        break;

    case ModuleHist:
        if(selection != nullptr) {
            histCanvas->UpdateHist(1, selection->GetHist("PHYS"));
            histCanvas->UpdateHist(2, selection->GetHist("LMS"));
            histCanvas->UpdateHist(3, selection->GetHist("PED"));
        }
        break;

     case TaggerHist:
         histCanvas->UpdateHist(1, handler->GetTagEHist());
         histCanvas->UpdateHist(2, handler->GetTagTHist());
         histCanvas->UpdateHist(3, handler->GetEnergyHist());
     }
}

void PRadEventViewer::SelectModule(HyCalModule* module)
{
    selection = module;
    UpdateHistCanvas();
    UpdateStatusInfo();
}

void PRadEventViewer::UpdateStatusInfo()
{
    if(selection == nullptr)
        return;

    QStringList valueList;
    QString typeInfo;

    ChannelAddress daqInfo = selection->GetDAQInfo();
    ChannelAddress hvInfo = selection->GetHVInfo();
    PRadDAQUnit::Geometry geoInfo = selection->GetGeometry();

    switch(geoInfo.type)
    {
    case HyCalModule::LeadTungstate:
        typeInfo = tr("<center><p><b>PbWO<sub>4</sub></b></p></center>");
        break;
    case HyCalModule::LeadGlass:
        typeInfo = tr("<center><p><b>Pb-Glass</b></p></center>");
        break;
    case HyCalModule::Scintillator:
        typeInfo = tr("<center><p><b>Scintillator</b></p></center>");
        break;
    case HyCalModule::LightMonitor:
        typeInfo = tr("<center><p><b>Light Monitor</b></p></center>");
        break;
    default:
        typeInfo = tr("<center><p><b>Unknown</b></p></center>");
        break;
    }

    // first value column
    valueList << selection->GetReadID()                                   // module ID
              << typeInfo                                                 // module type
              << tr("C") + QString::number(daqInfo.crate)                 // daq crate
                 + tr(", S") + QString::number(daqInfo.slot)              // daq slot
                 + tr(", Ch") + QString::number(daqInfo.channel)          // daq channel
              << QString::fromStdString(selection->GetTDCName())          // tdc group
              << tr("C") + QString::number(hvInfo.crate)                  // hv crate
                 + tr(", S") + QString::number(hvInfo.slot)               // hv slot
                 + tr(", Ch") + QString::number(hvInfo.channel)           // hv channel
              << QString::number(selection->GetOccupancy());              // Occupancy

    PRadDAQUnit::Pedestal ped = selection->GetPedestal();

#ifdef USE_CAEN_HV
    PRadHVSystem::Voltage volt = hvSystem->GetVoltage(hvInfo.crate, hvInfo.slot, hvInfo.channel);
    QString temp = QString::number(volt.Vmon) + tr(" V ")
                   + ((volt.ON)? tr("/ ") : tr("(OFF) / "))
                   + QString::number(volt.Vset) + tr(" V");
#else
    QString temp = "N/A";
#endif

    // second value column
    valueList << QString::number(ped.mean)                                // pedestal mean
#if QT_VERSION >= 0x050000
                 + tr(" \u00B1 ")
#else
                 + tr(" \261 ")
#endif
                 + QString::number(ped.sigma,'f',2)                       // pedestal sigma
              << QString::number(handler->GetCurrentEventNb())            // current event
              << QString::number(selection->GetEnergy()) + tr(" MeV / ")  // energy
                 + QString::number(handler->GetEnergy()) + tr(" MeV")     // total energy
              << QString::number(selection->GetADC())                     // ADC value
              << temp                                                     // HV info
              << QString::number(selection->GetCustomValue());            // custom value

    // update status info window
    for(int i = 0; i < 6; ++i)
    {
        statusItem[i]->setText(1, valueList.at(i));
        statusItem[i]->setText(3, valueList.at(6+i));
    }
}

void PRadEventViewer::readEventFromFile(const QString &filepath)
{
    std::cout << "Reading data from file " << filepath.toStdString() << std::endl;
#ifdef USE_EVIO_LIB
    try {
        evio::evioFileChannel *chan = new evio::evioFileChannel(filepath.toStdString().c_str(),"r");
        chan->open();

        while(chan->read())
        {
            handler->Decode(chan->getBuffer());
        }

        chan->close();
        delete chan;

    } catch (evio::evioException e) {
        std::cerr << e.toString() << endl;
    } catch (...) {
        std::cerr << "?unknown exception" << endl;
    }
#else
    handler->ReadFromEvio(filepath.toStdString());
#endif
}

void PRadEventViewer::readCustomValue(const QString &filepath)
{
    ConfigParser c_parser;

    if(!c_parser.OpenFile(filepath.toStdString())) {
        std::cerr << "Cannot open custom map file "
                  << "\"" << filepath.toStdString() << "\"."
                  << std::endl;
        return;
    }

    ModuleAction(&HyCalModule::UpdateCustomValue, 0.);

    double min_value = 0.;
    double max_value = 1.;

    while(c_parser.ParseLine())
    {
        if(!c_parser.NbofElements())
            continue;

        if(c_parser.NbofElements() == 2) {
            std::string name;
            double value;
            c_parser >> name >> value;
            HyCalModule *module = dynamic_cast<HyCalModule*>(handler->GetChannel(name));
            if(module != nullptr) {
                module->UpdateCustomValue(value);
                min_value = std::min(value, min_value);
                max_value = std::max(value, max_value);
            }
        } else {
            std::cout << "Unrecognized custom map format, skipped one line." << std::endl;
        }

    }

    viewModeBox->setCurrentIndex((int)CustomView);

    specSetting->SetSpectrumRange(floor(min_value*1.3), floor(max_value*1.3));
    specSetting->SetLinearScale();
    Refresh();
}


QString PRadEventViewer::getFileName(const QString &title,
                                     const QString &dir,
                                     const QStringList &filter,
                                     const QString &suffix,
                                     QFileDialog::AcceptMode mode)
{
    QFileDialog::FileMode fmode = QFileDialog::ExistingFile;
    if(mode == QFileDialog::AcceptSave)
        fmode =QFileDialog::AnyFile;

    QStringList filepaths = getFileNames(title, dir, filter, suffix, mode, fmode);
    if(filepaths.size())
        return filepaths.at(0);

    return "";
}

QStringList PRadEventViewer::getFileNames(const QString &title,
                                          const QString &dir,
                                          const QStringList &filter,
                                          const QString &suffix,
                                          QFileDialog::AcceptMode mode,
                                          QFileDialog::FileMode fmode)
{
    QStringList filepath;
    fileDialog->setWindowTitle(title);
    fileDialog->setDirectory(dir);
    fileDialog->setNameFilters(filter);
    fileDialog->setDefaultSuffix(suffix);
    fileDialog->setAcceptMode(mode);
    fileDialog->setFileMode(fmode);

    if(fileDialog->exec())
        filepath = fileDialog->selectedFiles();

    return filepath;
}

void PRadEventViewer::saveHistToFile()
{
    QString rootFile = getFileName(tr("Save histograms to root file"),
                                   tr("rootfiles/"),
                                   QStringList(tr("root files (*.root)")),
                                   tr("root"),
                                   QFileDialog::AcceptSave);

    if(rootFile.isEmpty()) // did not open a file
        return;

    handler->SaveHistograms(rootFile.toStdString());

    rStatusLabel->setText(tr("All histograms are saved to ") + rootFile);
}

void PRadEventViewer::savePedestalFile()
{
    QString pedFile = getFileName(tr("Save pedestal to file"),
                                  tr("config/"),
                                  QStringList(tr("data files (*.dat)")),
                                  tr("dat"),
                                  QFileDialog::AcceptSave);
    if(pedFile.isEmpty())
        return;

    std::ofstream pedestalmap(pedFile.toStdString());

    for(auto &channel : handler->GetChannelList())
    {
        ChannelAddress daqInfo = channel->GetDAQInfo();
        PRadDAQUnit::Pedestal ped = channel->GetPedestal();
        pedestalmap << daqInfo.crate << "  "
                    << daqInfo.slot << "  "
                    << daqInfo.channel << "  "
                    << ped.mean << "  "
                    << ped.sigma << std::endl;
    }

    pedestalmap.close();
}

void PRadEventViewer::findPeak()
{
    if(selection == nullptr) return;
    TH1 *h = selection->GetHist("PHYS");
    //Use TSpectrum to find the peak candidates
    TSpectrum s(10);
    int nfound = s.Search(h, 20 , "", 0.05);
    if(nfound) {
        double ped = selection->GetPedestal().mean;
        auto *xpeaks = s.GetPositionX();
        std::cout <<"Main peak location: " << xpeaks[0] <<". "
                  << int(xpeaks[0] - ped) << " away from the pedestal."
                  << std:: endl;
        UpdateHistCanvas();
    }
}

void PRadEventViewer::fitPedestal()
{
    handler->FitPedestal();
    UpdateHistCanvas();
    emit currentEventChanged(eventSpin->value());
}

void PRadEventViewer::fitHistogram()
{
    QDialog dialog(this);
    // Use a layout allowing to have a label next to each field
    QFormLayout form(&dialog);

    // Add some text above the fields
    form.addRow(new QLabel("Select histogram and range:"));

    // Add the lineEdits with their respective labels
    QVector<QLineEdit *> fields;
    QStringList label, de_value;

    label << tr("Channel")
          << tr("Histogram Name")
          << tr("Fitting Function (root format)")
          << tr("Range Min.")
          << tr("Range Max.");

    de_value << ((selection) ? QString::fromStdString(selection->GetName()) : "W1")
             << "PHYS"
             << "gaus"
             << "0"
             << "8000";

    for(int i = 0; i < 5; ++i)
    {
        QLineEdit *lineEdit = new QLineEdit(&dialog);
        lineEdit->setText(de_value.at(i));
        form.addRow(label.at(i), lineEdit);
        fields.push_back(lineEdit);
    }

    // Add some standard buttons (Cancel/Ok) at the bottom of the dialog
    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                               Qt::Horizontal, &dialog);
    form.addRow(&buttonBox);
    QObject::connect(&buttonBox, SIGNAL(accepted()), &dialog, SLOT(accept()));
    QObject::connect(&buttonBox, SIGNAL(rejected()), &dialog, SLOT(reject()));

    // Show the dialog as modal
    if (dialog.exec() == QDialog::Accepted) {
        // If the user didn't dismiss the dialog, do something with the fields
        try {
            auto pars = handler->FitHistogram(fields.at(0)->text().toStdString(),
                                              fields.at(1)->text().toStdString(),
                                              fields.at(2)->text().toStdString(),
                                              fields.at(3)->text().toDouble(),
                                              fields.at(4)->text().toDouble(),
                                              true);

            UpdateHistCanvas();

        } catch (PRadException e) {
            QMessageBox::critical(this,
                                  QString::fromStdString(e.FailureType()),
                                  QString::fromStdString(e.FailureDesc()));

        }
    }
}

void PRadEventViewer::correctGainFactor()
{
    QRegExp reg("[0-9]{6}");
    if(reg.indexIn(fileName) != -1) {
        int run_number = reg.cap(0).toInt();
        handler->SetRunNumber(run_number);
    }

    handler->CorrectGainFactor();
    // Refill the histogram to show the changes
    handler->RefillEnergyHist();
    UpdateHistCanvas();
    emit currentEventChanged(eventSpin->value());
}

void PRadEventViewer::takeSnapShot()
{

#if QT_VERSION >= 0x050000
    QPixmap p = QGuiApplication::primaryScreen()->grabWindow(QApplication::activeWindow()->winId(), 0, 0);
#else
    QPixmap p = QPixmap::grabWindow(QApplication::activeWindow()->winId());
#endif

    // using date time as file name
    QString datetime = tr("snapshots/") + QDateTime::currentDateTime().toString();
    datetime.replace(QRegExp("\\s+"), "_");

    QString filepath = datetime + tr(".png");

    // make sure no snapshots are overwritten
    int i = 0;
    while(1) {
        QFileInfo check(filepath);
        if(!check.exists())
            break;
        ++i;
        filepath = datetime + tr("_") + QString::number(i) + tr(".png");
    }

    p.save(filepath);

    // update info
    rStatusLabel->setText(tr("Snap shot saved to ") + filepath);
}

void PRadEventViewer::editCustomValueLabel(QTreeWidgetItem* item, int column)
{
    if(item == statusItem[5] && column == 2)
        item->setFlags(item->flags() | Qt::ItemIsEditable);
    else
        item->setFlags(item->flags() & ~Qt::ItemIsEditable);
}

void PRadEventViewer::handleRootEvents()
{
    gSystem->ProcessEvents();
}

#ifdef RECON_DISPLAY
//============================================================================//
// Reconstruction Display functions                                           //
//============================================================================//

void PRadEventViewer::setupReconDisplay()
{
    // load GEM configuration
    handler->ReadGEMConfiguration("config/gem_map.conf");
    handler->ReadGEMPedestalFile("config/gem_ped.dat");

    // add hycal clustering methods
    coordSystem = new PRadCoordSystem("config/coordinates.dat");
    detMatch = new PRadDetMatch("config/det_match.conf");

    reconSetting = new ReconSettingPanel(this);
    reconSetting->ConnectDataHandler(handler);
    reconSetting->ConnectCoordSystem(coordSystem);
    reconSetting->ConnectMatchSystem(detMatch);

}

QMenu *PRadEventViewer::setupReconMenu()
{
    QMenu *reconMenu = new QMenu(tr("&Reconstruct Event"));

    enableRecon = reconMenu->addAction(tr("Show Reconstructed Event"));
    enableRecon->setCheckable(true);
    enableRecon->setChecked(true);

    QAction *setupRecon = reconMenu->addAction(tr("Reconstruction Settings"));
    setupRecon->setShortcut(QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_R));

    connect(enableRecon, SIGNAL(triggered()), this, SLOT(enableReconstruct()));
    connect(setupRecon, SIGNAL(triggered()), this, SLOT(setupReconMethods()));

    return reconMenu;
}

void PRadEventViewer::enableReconstruct()
{
    if(!enableRecon->isChecked())
        HyCal->ClearHitsMarks();

    emit(changeCurrentEvent(eventSpin->value()));
}

void PRadEventViewer::setupReconMethods()
{
    // sync settings with the connected objects
    reconSetting->SyncSettings();

    // save for restore
    reconSetting->SaveSettings();

    if(!reconSetting->exec()) {
        reconSetting->RestoreSettings();
        return;
    }

    // apply the changes to connected objects
    reconSetting->ApplyChanges();

    emit(changeCurrentEvent(eventSpin->value()));
}

void PRadEventViewer::showReconEvent(int evt)
{
    HyCal->ClearHitsMarks();
    if(handler->GetEventCount() == 0)
        return;

    auto &thisEvent = handler->GetEvent(evt);

    if(!thisEvent.is_physics_event())
        return;

    // reconstruction
    PRadGEMSystem *gem_srs = handler->GetSRS();
    handler->HyCalReconstruct(thisEvent);
    gem_srs->Reconstruct(thisEvent);

    // get reconstructed clusters
    int n, n1, n2;
    HyCalHit *hycal_hit = handler->GetHyCalCluster(n);
    GEMHit *gem1_hit = gem_srs->GetDetector("PRadGEM1")->GetCluster(n1);
    GEMHit *gem2_hit = gem_srs->GetDetector("PRadGEM2")->GetCluster(n2);

    // coordinates transform, projection
    coordSystem->Transform(hycal_hit, n);
    coordSystem->Transform(gem1_hit, n1);
    coordSystem->Transform(gem2_hit, n2);

    coordSystem->Projection(hycal_hit, n);
    coordSystem->Projection(gem1_hit, n1);
    coordSystem->Projection(gem2_hit, n2);

    // hits matching
    auto matched = detMatch->Match(hycal_hit, n, gem1_hit, n1, gem2_hit, n2);

    // display HyCal hits
    if(reconSetting->ShowDetector(PRadDetectors::HyCal)) {

        HyCalScene::MarkAttributes attr = reconSetting->GetMarkAttributes(PRadDetectors::HyCal);
        if(reconSetting->ShowMatchedDetector(PRadDetectors::HyCal)) {
            for(auto &m : matched)
            {
                QPointF p(CARTESIAN_TO_HYCALSCENE(hycal_hit[m.hycal].x, hycal_hit[m.hycal].y));
                HyCal->AddHitsMark("HyCal Hit", p, attr, QString::number(hycal_hit[m.hycal].E) + "MeV");
            }
        } else {
            for(int i = 0; i < n; ++i)
            {
                QPointF p(CARTESIAN_TO_HYCALSCENE(hycal_hit[i].x, hycal_hit[i].y));
                HyCal->AddHitsMark("HyCal Hit", p, attr, QString::number(hycal_hit[i].E) + " MeV");
            }
        }

    }

    // display GEM1 hits
    if(reconSetting->ShowDetector(PRadDetectors::PRadGEM1)) {

        HyCalScene::MarkAttributes attr = reconSetting->GetMarkAttributes(PRadDetectors::PRadGEM1);
        if(reconSetting->ShowMatchedDetector(PRadDetectors::PRadGEM1)) {
            for(auto &m : matched)
            {
                if(m.gem1 != -1) {
                    QPointF p(CARTESIAN_TO_HYCALSCENE(gem1_hit[m.gem1].x, gem1_hit[m.gem1].y));
                    HyCal->AddHitsMark("GEM1 Hit", p, attr);
                }
            }
        } else {
            for(int i = 0; i < n1; ++i)
            {
                QPointF p(CARTESIAN_TO_HYCALSCENE(gem1_hit[i].x, gem1_hit[i].y));
                HyCal->AddHitsMark("GEM1 Hit", p, attr);
            }
        }
    }

    // display GEM2 hits
    if(reconSetting->ShowDetector(PRadDetectors::PRadGEM1)) {

        HyCalScene::MarkAttributes attr = reconSetting->GetMarkAttributes(PRadDetectors::PRadGEM2);
        if(reconSetting->ShowMatchedDetector(PRadDetectors::PRadGEM2)) {
            for(auto &m : matched)
            {
                if(m.gem2 != -1) {
                    QPointF p(CARTESIAN_TO_HYCALSCENE(gem2_hit[m.gem2].x, gem2_hit[m.gem2].y));
                    HyCal->AddHitsMark("GEM2 Hit", p, attr);
                }
            }
        } else {
            for(int i = 0; i < n2; ++i)
            {
                QPointF p(CARTESIAN_TO_HYCALSCENE(gem2_hit[i].x, gem2_hit[i].y));
                HyCal->AddHitsMark("GEM2 Hit", p, attr);
            }
        }
    }

    // display hits
    Refresh();
}
#endif

#ifdef USE_ONLINE_MODE
//============================================================================//
// Online mode functions                                                      //
//============================================================================//

void PRadEventViewer::setupOnlineMode()
{
    etSetting = new ETSettingPanel(this);
    onlineTimer = new QTimer(this);
    connect(onlineTimer, SIGNAL(timeout()), this, SLOT(handleOnlineTimer()));
    // future watcher for online mode
    connect(&watcher, SIGNAL(finished()), this, SLOT(startOnlineMode()));

    etChannel = new PRadETChannel();
}

QMenu *PRadEventViewer::setupOnlineMenu()
{
    // online menu, toggle on/off online mode
    QMenu *onlineMenu = new QMenu(tr("Online &Mode"));

    onlineEnAction = onlineMenu->addAction(tr("Start Online Mode"));
    onlineDisAction = onlineMenu->addAction(tr("Stop Online Mode"));
    onlineDisAction->setEnabled(false);

    connect(onlineEnAction, SIGNAL(triggered()), this, SLOT(initOnlineMode()));
    connect(onlineDisAction, SIGNAL(triggered()), this, SLOT(stopOnlineMode()));

    return onlineMenu;
}

void PRadEventViewer::initOnlineMode()
{
    if(!etSetting->exec())
        return;

    // Disable buttons
    onlineEnAction->setEnabled(false);
    openDataAction->setEnabled(false);
    eventSpin->setEnabled(false);
    future = QtConcurrent::run(this, &PRadEventViewer::connectETClient);
    watcher.setFuture(future);
}

bool PRadEventViewer::connectETClient()
{
    try {
        etChannel->Open(etSetting->GetETHost().toStdString().c_str(),
                        etSetting->GetETPort(),
                        etSetting->GetETFilePath().toStdString().c_str());
        etChannel->NewStation(etSetting->GetStationName().toStdString());
        etChannel->AttachStation();
    } catch(PRadException e) {
        etChannel->ForceClose();
        std::cerr << e.FailureType() << ": "
                  << e.FailureDesc() << std::endl;
        return false;
    }

    return true;
}

void PRadEventViewer::startOnlineMode()
{
    if(!future.result()) { // did not connected to ET
        QMessageBox::critical(this,
                              "Online Mode",
                              "Failure in Open&Attach to ET!");

        rStatusLabel->setText(tr("Failed to start Online Mode!"));
        onlineEnAction->setEnabled(true);
        openDataAction->setEnabled(true);
        eventSpin->setEnabled(true);
        return;
    }

    QMessageBox::information(this,
                             tr("Online Mode"),
                             tr("Online Monitor Start!"));

    onlineDisAction->setEnabled(true);
    // Successfully attach to ET, change to online mode
    handler->SetOnlineMode(true);

    // Clean buffer
    eraseData();

    // Update to status bar
    UpdateStatusBar(ONLINE_MODE);

    // show scalar counts
    HyCal->ShowScalers(true);
    Refresh();

    // Start online timer
    onlineTimer->start(5000);
}

void PRadEventViewer::stopOnlineMode()
{
    // Stop timer
    onlineTimer->stop();

    etChannel->ForceClose();
    QMessageBox::information(this,
                             tr("Online Monitor"),
                             tr("Dettached from ET!"));

    handler->SetOnlineMode(false);

    // Enable buttons
    onlineEnAction->setEnabled(true);
    openDataAction->setEnabled(true);
    onlineDisAction->setEnabled(false);
    eventSpin->setEnabled(true);

    // Update to Main Window
    UpdateStatusBar(NO_INPUT);

    // turn off show scalars
    HyCal->ShowScalers(false);
    Refresh();
}

void PRadEventViewer::handleOnlineTimer()
{
//   QtConcurrent::run(this, &PRadEventViewer::onlineUpdate, ET_CHUNK_SIZE);
    onlineUpdate(ET_CHUNK_SIZE);
}

void PRadEventViewer::onlineUpdate(const size_t &max_events)
{
    try {
        size_t num;

        for(num = 0; etChannel->Read() && num < max_events; ++num)
        {
            handler->Decode(etChannel->GetBuffer());
        }

        if(num) {
            UpdateHistCanvas();
            UpdateOnlineInfo();
            Refresh();
        }

    } catch(PRadException e) {
        std::cerr << e.FailureType() << ": "
                  << e.FailureDesc() << std::endl;
        return;
    }
}

void PRadEventViewer::UpdateOnlineInfo()
{
    QStringList onlineText;
    auto info = handler->GetOnlineInfo();

    for(auto &trg : info.trigger_info)
    {
        onlineText << QString::number(trg.freq) + tr(" Hz");
    }

    onlineText << QString::number(info.live_time*100.) + tr("%");
    onlineText << QString::number(info.beam_current) + tr(" nA");

    HyCal->UpdateScalerBox(onlineText);
}
#endif

#ifdef USE_CAEN_HV
//============================================================================//
// high voltage control functions                                             //
//============================================================================//

void PRadEventViewer::setupHVSystem()
{
    connect(this, SIGNAL(HVSystemInitialized()), this, SLOT(startHVMonitor()));

    hvSystem = new PRadHVSystem(this);

    QFile hvCrateList("config/hv_crate_list.txt");

    if(!hvCrateList.open(QFile::ReadOnly | QFile::Text)) {
        std::cout << "WARNING: Missing HV crate list"
                  << "\" config/hv_crate_list.txt \", "
                  << "no HV crate added!"
                  << std::endl;
        return;
    }

    std::string name, ip;
    int id;

    QTextStream in(&hvCrateList);

    while(!in.atEnd())
    {
        QString line = in.readLine().simplified();
        if(line.at(0) == '#')
            continue;
        QStringList fields = line.split(QRegExp("\\s+"));
        if(fields.size() == 3) {
            name = fields.takeFirst().toStdString();
            ip = fields.takeFirst().toStdString();
            id = fields.takeFirst().toInt();
            hvSystem->AddCrate(name, ip, id);
        }
    }

    hvCrateList.close();
}

QMenu *PRadEventViewer::setupHVMenu()
{
    // high voltage menu
    QMenu *hvMenu = new QMenu(tr("High &Voltage"));
    hvEnableAction = hvMenu->addAction(tr("Connect to HV system"));
    hvDisableAction = hvMenu->addAction(tr("Disconnect to HV system"));
    hvDisableAction->setEnabled(false);
    hvSaveAction = hvMenu->addAction(tr("Save HV Setting"));
    hvSaveAction->setEnabled(false);
    hvRestoreAction = hvMenu->addAction(tr("Restore HV Setting"));
    hvRestoreAction->setEnabled(false);

    connect(hvEnableAction, SIGNAL(triggered()), this, SLOT(connectHVSystem()));
    connect(hvDisableAction, SIGNAL(triggered()), this, SLOT(disconnectHVSystem()));
    connect(hvSaveAction, SIGNAL(triggered()), this, SLOT(saveHVSetting()));
    connect(hvRestoreAction, SIGNAL(triggered()), this, SLOT(restoreHVSetting()));

    return hvMenu;
}

void PRadEventViewer::connectHVSystem()
{
    hvEnableAction->setEnabled(false);
    hvDisableAction->setEnabled(false);
    hvSaveAction->setEnabled(false);
    hvRestoreAction->setEnabled(false);
    QtConcurrent::run(this, &PRadEventViewer::initHVSystem);
}

void PRadEventViewer::initHVSystem()
{
    hvSystem->Connect();
    emit HVSystemInitialized();
}

void PRadEventViewer::startHVMonitor()
{
    hvSystem->StartMonitor();
    hvDisableAction->setEnabled(true);
    hvSaveAction->setEnabled(true);
    hvRestoreAction->setEnabled(true);
}

void PRadEventViewer::disconnectHVSystem()
{
    hvSystem->Disconnect();
    hvEnableAction->setEnabled(true);
    hvDisableAction->setEnabled(false);
    hvSaveAction->setEnabled(false);
    hvRestoreAction->setEnabled(false);
    Refresh();
}

void PRadEventViewer::saveHVSetting()
{
    QString hvFile = getFileName(tr("Save High Voltage Settings to file"),
                                 tr("high_voltage/"),
                                 QStringList(tr("text files (*.txt)")),
                                 tr("txt"),
                                 QFileDialog::AcceptSave);

    if(hvFile.isEmpty()) // did not open a file
        return;

    hvSystem->StopMonitor();
    hvSystem->SaveCurrentSetting(hvFile.toStdString());
    hvSystem->StartMonitor();
}

void PRadEventViewer::restoreHVSetting()
{
    QString hvFile = getFileName(tr("Restore High Voltage Settings from file"),
                                 tr("high_voltage/"),
                                 QStringList(tr("text files (*.txt)")),
                                 tr("txt"),
                                 QFileDialog::AcceptOpen);

    if(hvFile.isEmpty()) // did not open a file
        return;

    hvSystem->StopMonitor();
    hvSystem->RestoreSetting(hvFile.toStdString());
    hvSystem->StartMonitor();
}
#endif
