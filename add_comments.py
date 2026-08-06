files = [
    "apps/laamba-governor/src/App.tsx",
    "apps/laamba-governor/src/panels/FormulaEditor.tsx",
    "apps/laamba-governor/src/panels/PipelineMixer.tsx",
    "apps/laamba-governor/src/panels/EngineRack.tsx",
    "apps/laamba-governor/src/panels/ConsolePanel.tsx",
    "apps/laamba-governor/src/panels/SampleBay.tsx",
    "apps/laamba-governor/src/panels/TopologyScope.tsx",
    "apps/laamba-governor/src/panels/ParameterRoll.tsx",
    "apps/laamba-governor/src/panels/ExperimentTimeline.tsx",
]

for f in files:
    with open(f, 'r') as file:
        content = file.read()

    if "Performance optimization" in content:
        continue

    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if "useStore(useShallow" in line:
            new_lines.append("  // Performance optimization: Using useShallow prevents unnecessary component re-renders")
            new_lines.append("  // by only subscribing to the specific store properties destructured below.")
        new_lines.append(line)

    with open(f, 'w') as file:
        file.write('\n'.join(new_lines))

with open('apps/laamba-governor/src/components/Toolbar.tsx', 'r') as file:
    content = file.read()
    if "Performance optimization" not in content:
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if "const {" in line and "selectedDataset:" in lines[lines.index(line)+1]:
                new_lines.append("  // Performance optimization: Using useShallow prevents unnecessary component re-renders")
                new_lines.append("  // by only subscribing to the specific store properties destructured below.")
            new_lines.append(line)
        with open('apps/laamba-governor/src/components/Toolbar.tsx', 'w') as file:
            file.write('\n'.join(new_lines))
