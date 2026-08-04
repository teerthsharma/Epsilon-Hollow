import { useEffect } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { invoke } from "@tauri-apps/api/core";
import { useStore } from "./store";
import SampleBay from "./panels/SampleBay";
import EngineRack from "./panels/EngineRack";
import PipelineMixer from "./panels/PipelineMixer";
import ExperimentTimeline from "./panels/ExperimentTimeline";
import ParameterRoll from "./panels/ParameterRoll";
import TopologyScope from "./panels/TopologyScope";
import ConsolePanel from "./panels/ConsolePanel";
import Toolbar from "./components/Toolbar";

export default function App() {
  const { setDatasets, addLog } = useStore();

  useEffect(() => {
    (async () => {
      try {
        addLog("Scanning datasets...");
        const result: any = await invoke("scan_datasets", { paths: [] });
        const ds = result.datasets || [];
        setDatasets(ds);
        addLog(`Found ${ds.length} datasets`);
      } catch (e: any) {
        addLog(`Dataset scan failed: ${e}`);
      }
    })();
  }, []);

  return (
    <div className="h-screen w-screen flex flex-col bg-gov-bg text-gov-text font-sans select-none">
      <Toolbar />

      <div className="flex-1 flex overflow-hidden">
        <PanelGroup direction="horizontal">
          <Panel defaultSize={15} minSize={10} className="flex flex-col">
            <SampleBay />
          </Panel>

          <PanelResizeHandle className="w-1 bg-gov-border hover:bg-gov-accent transition-colors" />

          <Panel defaultSize={55} minSize={30} className="flex flex-col">
            <PanelGroup direction="vertical">
              <Panel defaultSize={55} minSize={20} className="flex flex-col">
                <PipelineMixer />
              </Panel>
              <PanelResizeHandle className="h-1 bg-gov-border hover:bg-gov-accent transition-colors" />
              <Panel defaultSize={45} minSize={15} className="flex flex-col">
                <TopologyScope />
              </Panel>
            </PanelGroup>
          </Panel>

          <PanelResizeHandle className="w-1 bg-gov-border hover:bg-gov-accent transition-colors" />

          <Panel defaultSize={30} minSize={15} className="flex flex-col">
            <PanelGroup direction="vertical">
              <Panel defaultSize={35} minSize={15} className="flex flex-col">
                <EngineRack />
              </Panel>
              <PanelResizeHandle className="h-1 bg-gov-border hover:bg-gov-accent transition-colors" />
              <Panel defaultSize={30} minSize={10} className="flex flex-col">
                <ParameterRoll />
              </Panel>
              <PanelResizeHandle className="h-1 bg-gov-border hover:bg-gov-accent transition-colors" />
              <Panel defaultSize={35} minSize={15} className="flex flex-col">
                <ExperimentTimeline />
              </Panel>
            </PanelGroup>
          </Panel>
        </PanelGroup>
      </div>

      <ConsolePanel />
    </div>
  );
}
