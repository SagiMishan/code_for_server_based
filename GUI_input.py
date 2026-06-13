
import os, shutil
# import customtkinter as ctk
# from tkinter import filedialog, simpledialog, messagebox

# ctk.set_appearance_mode("dark")
# ctk.set_default_color_theme("blue")

def save_params_to_file(params, file_path):
    try:
        with open(file_path, "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
        print(f"Parameters saved to {file_path}")
    except Exception as e:
        print(f"Save Error: {e}")

def parse_params_txt(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or ":" not in line: continue
            k, v = line.split(':', 1)
            k, v = k.strip(), v.strip()
            if v.lower() in {'true', 'false'}: v = v.lower() == 'true'
            elif v.startswith('[') and v.endswith(']'): v = eval(v)
            else:
                try: v = int(v)
                except:
                    try: v = float(v)
                    except: pass
            out[k] = v
    return out

def get_folder_names():
    return [d for d in os.listdir() if os.path.isdir(d)]

# def get_parameters_gui_grouped():
    root = ctk.CTk()
    root.title("Model Parameters")
    results = {}

    # Helper: labeled entry
    def make_entry(parent, label, default, row, col_offset=0):
        ctk.CTkLabel(parent, text=label).grid(row=row, column=col_offset, padx=6, pady=3, sticky="e")
        e = ctk.CTkEntry(parent, width=120)
        e.insert(0, default)
        e.grid(row=row, column=col_offset+1, padx=6, pady=3, sticky="w")
        return e

    # Helper: labeled combobox
    def make_combo(parent, label, values, default, row, col_offset=0):
        ctk.CTkLabel(parent, text=label).grid(row=row, column=col_offset, padx=6, pady=3, sticky="e")
        c = ctk.CTkComboBox(parent, values=values, width=120)
        c.set(default)
        c.grid(row=row, column=col_offset+1, padx=6, pady=3, sticky="w")
        return c

    # ── General column ──────────────────────────────────────────────────────
    frame_general = ctk.CTkScrollableFrame(root, label_text="General / Init Parameters", width=300, height=560)
    frame_general.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
    rg = 0

    # Model name + copy button
    ctk.CTkLabel(frame_general, text="Name_of_model").grid(row=rg, column=0, padx=6, pady=3, sticky="e")
    folders = get_folder_names()
    name_var = ctk.CTkComboBox(frame_general, values=folders, width=120)
    name_var.set(folders[0] if folders else "")
    name_var.grid(row=rg, column=1, padx=6, pady=3, sticky="w")
    copy_btn = ctk.CTkButton(frame_general, text="Copy settings", state="disabled", width=110)
    copy_btn.grid(row=rg, column=2, padx=4, pady=3)
    rg += 1

    def fill_entry(e, val):
        e.delete(0, "end"); e.insert(0, str(val))

    def fill_gui(params):
        start_from_zero_var.set(int(params.get("start_from_zero", 0)))
        random_network_var.set(int(params.get("random_network", 1)))
        does_stage_1_var.set(int(params.get("does_stage_1", 1)))
        does_stage_2_var.set(int(params.get("does_stage_2", 1)))
        does_stage_3_var.set(int(params.get("does_stage_3", 1)))
        demod_type_var.set(params.get("demod_type", "complex"))
        fill_entry(N_transmitter_var, params.get("N_transmitter", 2))
        fill_entry(N_relays_var, params.get("N_relays", 30))
        fill_entry(N_users_var, ",".join(str(x) for x in params.get("N_users", [2, 2])))
        fill_entry(N_channels_var, params.get("N_channels", 2))
        fill_entry(N_rx_var, params.get("N_rx", 2))
        fill_entry(N_tx_var, params.get("N_tx", 1))
        fill_entry(d_spacing_var, params.get("d_spacing", 0.01))
        fill_entry(SNR_basic_trainning_var, params.get("SNR_basic_trainning", 50))
        fill_entry(SNR_max_var, params.get("SNR_max", 30))
        fill_entry(max_iteration_var, params.get("max_iteration", 10))
        fill_entry(SNR_step_var, params.get("SNR_step", 1))
        fill_entry(BER_th_var, params.get("BER_th", 1e-3))
        fill_entry(epochs_var, params.get("epochs", 1))
        z0_type_var.set(params.get("z0_type", "exp"))
        fill_entry(z0_init_var, params.get("z0_init", 0.1))
        fill_entry(z0_end_var, params.get("z0_end", 0.1))
        fill_entry(B_var, params.get("B", 10))
        fill_entry(sub_stages_var, ",".join(str(x) for x in params.get("sub_stages", [0.4, 0.5, 0.1])))

    def on_name_select(choice=None):
        folder = name_var.get()
        p = os.path.join(folder, "parameters.txt")
        if os.path.exists(p):
            copy_btn.configure(state="normal", command=lambda: fill_gui(parse_params_txt(p)))
        else:
            copy_btn.configure(state="disabled")
    name_var.configure(command=on_name_select)

    def update_folderlist():
        folders = get_folder_names()
        name_var.configure(values=folders)
        if folders:
            name_var.set(folders[0])
            on_name_select()

    # Checkboxes
    start_from_zero_var = ctk.IntVar(value=0)
    ctk.CTkCheckBox(frame_general, text="start_from_zero", variable=start_from_zero_var).grid(
        row=rg, column=0, columnspan=2, padx=6, pady=2, sticky="w"); rg+=1
    random_network_var = ctk.IntVar(value=1)
    ctk.CTkCheckBox(frame_general, text="random_network", variable=random_network_var).grid(
        row=rg, column=0, columnspan=2, padx=6, pady=2, sticky="w"); rg+=1
    does_stage_1_var = ctk.IntVar(value=0)
    ctk.CTkCheckBox(frame_general, text="does_stage_1", variable=does_stage_1_var).grid(
        row=rg, column=0, columnspan=2, padx=6, pady=2, sticky="w"); rg+=1
    does_stage_2_var = ctk.IntVar(value=1)
    ctk.CTkCheckBox(frame_general, text="does_stage_2", variable=does_stage_2_var).grid(
        row=rg, column=0, columnspan=2, padx=6, pady=2, sticky="w"); rg+=1
    does_stage_3_var = ctk.IntVar(value=1)
    ctk.CTkCheckBox(frame_general, text="does_stage_3", variable=does_stage_3_var).grid(
        row=rg, column=0, columnspan=2, padx=6, pady=2, sticky="w"); rg+=1

    demod_type_var = make_combo(frame_general, "demod_type", ["simple", "complex"], "simple", rg); rg+=1
    N_transmitter_var = make_entry(frame_general, "N_transmitter", "2", rg); rg+=1
    N_relays_var      = make_entry(frame_general, "N_relays", "30", rg); rg+=1
    N_users_var       = make_entry(frame_general, "N_users (comma sep)", "2,2", rg); rg+=1
    N_channels_var    = make_entry(frame_general, "N_channels", "2", rg); rg+=1
    N_rx_var          = make_entry(frame_general, "N_rx (RX ant/user)", "2", rg); rg+=1
    N_tx_var          = make_entry(frame_general, "N_tx (TX ant/tx)", "1", rg); rg+=1
    d_spacing_var     = make_entry(frame_general, "ant spacing (m)", "0.1", rg); rg+=1

    # ── Stage 1 column ──────────────────────────────────────────────────────
    frame_stage1 = ctk.CTkFrame(root)
    frame_stage1.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")
    ctk.CTkLabel(frame_stage1, text="Stage 1 Parameters", font=ctk.CTkFont(size=14, weight="bold")).grid(
        row=0, column=0, columnspan=2, pady=(8, 4))
    r1 = 1
    SNR_basic_trainning_var = make_entry(frame_stage1, "SNR_basic_trainning", "50",  r1); r1+=1
    SNR_max_var             = make_entry(frame_stage1, "SNR_max",             "30",  r1); r1+=1
    max_iteration_var       = make_entry(frame_stage1, "max_iteration",       "10",  r1); r1+=1
    SNR_step_var            = make_entry(frame_stage1, "SNR_step",            "1",   r1); r1+=1
    BER_th_var              = make_entry(frame_stage1, "BER_th",              "1e-3",r1); r1+=1

    # ── Stage 2 column ──────────────────────────────────────────────────────
    frame_stage2 = ctk.CTkFrame(root)
    frame_stage2.grid(row=0, column=2, padx=8, pady=8, sticky="nsew")
    ctk.CTkLabel(frame_stage2, text="Stage 2 Parameters", font=ctk.CTkFont(size=14, weight="bold")).grid(
        row=0, column=0, columnspan=2, pady=(8, 4))
    r2 = 1
    epochs_var     = make_entry(frame_stage2, "epochs",                "1",           r2); r2+=1
    z0_type_var    = make_combo(frame_stage2, "z0_type", ["exp","const","linear"], "exp", r2); r2+=1
    z0_init_var    = make_entry(frame_stage2, "z0_init",               "0.1",         r2); r2+=1
    z0_end_var     = make_entry(frame_stage2, "z0_end",                "0.1",         r2); r2+=1
    B_var          = make_entry(frame_stage2, "B",                     "10",          r2); r2+=1
    sub_stages_var = make_entry(frame_stage2, "sub_stages (comma sep)","0.4,0.5,0.1", r2); r2+=1

    # ── Bottom buttons ───────────────────────────────────────────────────────
    def clone_folder():
        src = filedialog.askdirectory(initialdir=os.getcwd(), title="Select folder to clone")
        if not src: return
        base = os.path.basename(src)
        new_name = simpledialog.askstring("Clone folder", f"Name for clone:", initialvalue=base+"_clone")
        if not new_name: return
        dst = os.path.join(os.getcwd(), new_name)
        if os.path.exists(dst):
            messagebox.showerror("Error", f"'{dst}' already exists."); return
        try:
            shutil.copytree(src, dst)
            messagebox.showinfo("Success", f"Cloned to '{dst}'")
            update_folderlist()
        except Exception as e:
            messagebox.showerror("Clone Error", str(e))

    ctk.CTkButton(root, text="Clone Folder", command=clone_folder).grid(
        row=1, column=0, columnspan=3, padx=8, pady=4, sticky="ew")

    def submit():
        try:
            results['Name_of_model']         = name_var.get()
            results['start_from_zero']        = bool(start_from_zero_var.get())
            results['random_network']         = bool(random_network_var.get())
            results['does_stage_1']           = bool(does_stage_1_var.get())
            results['does_stage_2']           = bool(does_stage_2_var.get())
            results['does_stage_3']           = bool(does_stage_3_var.get())
            results['demod_type']             = demod_type_var.get()
            results['N_transmitter']          = int(N_transmitter_var.get())
            results['N_relays']               = int(N_relays_var.get())
            results['N_users']                = [int(x) for x in N_users_var.get().split(',')]
            results['N_channels']             = int(N_channels_var.get())
            results['N_rx']                   = int(N_rx_var.get())
            results['N_tx']                   = int(N_tx_var.get())
            results['d_spacing']              = float(d_spacing_var.get())
            results['SNR_basic_trainning']    = float(SNR_basic_trainning_var.get())
            results['SNR_max']                = float(SNR_max_var.get())
            results['max_iteration']          = int(max_iteration_var.get())
            results['SNR_step']               = float(SNR_step_var.get())
            results['BER_th']                 = float(BER_th_var.get())
            results['epochs']                 = int(epochs_var.get())
            results['z0_type']                = z0_type_var.get()
            results['z0_init']                = float(z0_init_var.get())
            results['z0_end']                 = float(z0_end_var.get())
            results['B']                      = float(B_var.get())
            results['sub_stages']             = [float(x) for x in sub_stages_var.get().split(',')]
            root.quit(); root.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    ctk.CTkButton(root, text="Submit", command=submit, fg_color="green", hover_color="darkgreen").grid(
        row=2, column=0, columnspan=3, padx=8, pady=8, sticky="ew")

    root.mainloop()
    return results