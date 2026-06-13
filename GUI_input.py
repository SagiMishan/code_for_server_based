# def save_params_to_file(params, file_path):
#     try:
#         with open(file_path, "w") as f:
#             for key, value in params.items():
#                 f.write(f"{key}: {value}\n")
#         print(f"Parameters saved to {file_path}")
#     except Exception as e:
#         print(f"Save Error: {e}")
#
# # Example usage:
# # save_params_as_plain_txt(params, "params_output.txt")
# import os, tkinter as tk
# from tkinter import ttk, filedialog, simpledialog, messagebox
#
# def parse_params_txt(path):
#     result = {}
#     with open(path) as f:
#         for line in f:
#             if not line.strip() or ':' not in line: continue
#             k,v = line.split(':',1)
#             k,v = k.strip(), v.strip()
#             if v.lower() in ['true', 'false']:
#                 v = v.lower() == 'true'
#             elif v.startswith('[') and v.endswith(']'):
#                 v = eval(v)
#             else:
#                 try: v = int(v)
#                 except:
#                     try: v = float(v)
#                     except: pass
#             result[k]=v
#     return result
#
# def get_folder_names():
#     return [d for d in os.listdir() if os.path.isdir(d)]
#
# import os, shutil, tkinter as tk
# from tkinter import ttk, messagebox, filedialog, simpledialog
#
# def parse_params_txt(path):
#     out = {}
#     with open(path) as f:
#         for line in f:
#             if not line.strip() or ":" not in line: continue
#             k,v = line.split(':',1)
#             k, v = k.strip(), v.strip()
#             if v.lower() in {'true','false'}: v = v.lower() == 'true'
#             elif v.startswith('[') and v.endswith(']'): v = eval(v)
#             else:
#                 try: v = int(v)
#                 except:
#                     try: v = float(v)
#                     except: pass
#             out[k]=v
#     return out
#
# def get_folder_names():
#     return [d for d in os.listdir() if os.path.isdir(d)]
#
# def get_parameters_gui_grouped():
#     root = tk.Tk()
#     root.title("Model Parameters")
#     results = {}
#
#     # --- General ---
#     # Wrap in a canvas+scrollbar so the column doesn't get clipped when the
#     # window is taller than the screen.
#     outer_general = tk.Frame(root)
#     outer_general.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
#     canvas_gen = tk.Canvas(outer_general, borderwidth=0, highlightthickness=0)
#     scrollbar_gen = tk.Scrollbar(outer_general, orient="vertical", command=canvas_gen.yview)
#     canvas_gen.configure(yscrollcommand=scrollbar_gen.set)
#     scrollbar_gen.pack(side="right", fill="y")
#     canvas_gen.pack(side="left", fill="both", expand=True)
#     frame_general = tk.LabelFrame(canvas_gen, text="General / Init Parameters", padx=8, pady=8)
#     canvas_gen.create_window((0, 0), window=frame_general, anchor="nw")
#     def _on_frame_configure(event):
#         canvas_gen.configure(scrollregion=canvas_gen.bbox("all"))
#         canvas_gen.config(height=min(frame_general.winfo_reqheight(), 600))
#     frame_general.bind("<Configure>", _on_frame_configure)
#     row=0
#     tk.Label(frame_general, text="Name_of_model").grid(row=row, column=0)
#     name_var = ttk.Combobox(frame_general, values=get_folder_names())
#     folders = get_folder_names()
#     name_var.set(folders[0] if folders else "")
#     name_var.grid(row=row, column=1)
#     copy_btn = tk.Button(frame_general, text="Copy settings to GUI", state=tk.DISABLED)
#     copy_btn.grid(row=row, column=2)
#     row+=1
#     def fill_gui(params):
#         start_from_zero_var.set(int(params.get("start_from_zero",0)))
#         random_network_var.set(int(params.get("random_network",1)))
#         does_stage_1_var.set(int(params.get("does_stage_1",1)))
#         does_stage_2_var.set(int(params.get("does_stage_2",1)))
#         does_stage_3_var.set(int(params.get("does_stage_3",1)))
#         demod_type_var.set(params.get("demod_type","complex"))
#         N_transmitter_var.delete(0, tk.END); N_transmitter_var.insert(0, str(params.get("N_transmitter",2)))
#         N_relays_var.delete(0, tk.END); N_relays_var.insert(0, str(params.get("N_relays",30)))
#         N_users_var.delete(0, tk.END); N_users_var.insert(0, ",".join(str(x) for x in params.get("N_users",[2,2])))
#         N_channels_var.delete(0, tk.END); N_channels_var.insert(0, str(params.get("N_channels",2)))
#         N_rx_var.delete(0, tk.END); N_rx_var.insert(0, str(params.get("N_rx",2)))
#         N_tx_var.delete(0, tk.END); N_tx_var.insert(0, str(params.get("N_tx",1)))
#         d_spacing_var.delete(0, tk.END); d_spacing_var.insert(0, str(params.get("d_spacing",0.01)))
#         SNR_basic_trainning_var.delete(0, tk.END); SNR_basic_trainning_var.insert(0, str(params.get("SNR_basic_trainning",50)))
#         SNR_max_var.delete(0, tk.END); SNR_max_var.insert(0, str(params.get("SNR_max",30)))
#         max_iteration_var.delete(0, tk.END); max_iteration_var.insert(0, str(params.get("max_iteration",10)))
#         SNR_step_var.delete(0, tk.END); SNR_step_var.insert(0, str(params.get("SNR_step",1)))
#         BER_th_var.delete(0, tk.END); BER_th_var.insert(0, str(params.get("BER_th",1e-3)))
#         epochs_var.delete(0, tk.END); epochs_var.insert(0, str(params.get("epochs",1)))
#         z0_type_var.set(params.get("z0_type","exp"))
#         z0_init_var.delete(0, tk.END); z0_init_var.insert(0, str(params.get("z0_init",0.1)))
#         z0_end_var.delete(0, tk.END); z0_end_var.insert(0, str(params.get("z0_end",0.1)))
#         B_var.delete(0, tk.END); B_var.insert(0, str(params.get("B",10)))
#         sub_stages_var.delete(0, tk.END); sub_stages_var.insert(0, ",".join(str(x) for x in params.get("sub_stages",[0.4,0.5,0.1])))
#
#     def on_name_select(event=None):
#         folder = name_var.get()
#         p = os.path.join(folder, "parameters.txt")
#         if os.path.exists(p):
#             copy_btn.config(state=tk.NORMAL)
#             copy_btn.config(command=lambda: fill_gui(parse_params_txt(p)))
#         else:
#             copy_btn.config(state=tk.DISABLED)
#     name_var.bind("<<ComboboxSelected>>", on_name_select)
#
#     def update_folderlist():
#         folders = get_folder_names()
#         name_var['values'] = folders
#         if folders:
#             name_var.set(folders[0])
#             on_name_select()
#
#     start_from_zero_var = tk.IntVar(value=0)
#     tk.Checkbutton(frame_general, text="start_from_zero", variable=start_from_zero_var).grid(row=row, column=0, columnspan=2); row+=1
#     random_network_var = tk.IntVar(value=1)
#     tk.Checkbutton(frame_general, text="random_network", variable=random_network_var).grid(row=row, column=0, columnspan=2); row+=1
#     does_stage_1_var = tk.IntVar(value=0)
#     tk.Checkbutton(frame_general, text="does_stage_1", variable=does_stage_1_var).grid(row=row, column=0, columnspan=2); row+=1
#     does_stage_2_var = tk.IntVar(value=1)
#     tk.Checkbutton(frame_general, text="does_stage_2", variable=does_stage_2_var).grid(row=row, column=0, columnspan=2); row+=1
#     does_stage_3_var = tk.IntVar(value=1)
#     tk.Checkbutton(frame_general, text="does_stage_3", variable=does_stage_3_var).grid(row=row, column=0, columnspan=2); row+=1
#     tk.Label(frame_general, text="demod_type").grid(row=row, column=0)
#     demod_type_var = ttk.Combobox(frame_general, values=["simple", "complex"]); demod_type_var.set("simple")
#     demod_type_var.grid(row=row, column=1); row+=1
#     tk.Label(frame_general, text="N_transmitter").grid(row=row, column=0)
#     N_transmitter_var = tk.Entry(frame_general); N_transmitter_var.insert(0, "2")
#     N_transmitter_var.grid(row=row, column=1); row+=1
#     tk.Label(frame_general, text="N_relays").grid(row=row, column=0)
#     N_relays_var = tk.Entry(frame_general); N_relays_var.insert(0, "30")
#     N_relays_var.grid(row=row, column=1); row+=1
#     tk.Label(frame_general, text="N_users (comma sep)").grid(row=row, column=0)
#     N_users_var = tk.Entry(frame_general); N_users_var.insert(0, "2,2")
#     N_users_var.grid(row=row, column=1); row+=1
#     tk.Label(frame_general, text="N_channels").grid(row=row, column=0)
#     N_channels_var = tk.Entry(frame_general); N_channels_var.insert(0, "2")
#     N_channels_var.grid(row=row, column=1); row+=1
#     tk.Label(frame_general, text="N_rx (RX antennas/user)").grid(row=row, column=0)
#     N_rx_var = tk.Entry(frame_general); N_rx_var.insert(0, "2")
#     N_rx_var.grid(row=row, column=1); row+=1
#     tk.Label(frame_general, text="N_tx (TX antennas/transmitter)").grid(row=row, column=0)
#     N_tx_var = tk.Entry(frame_general); N_tx_var.insert(0, "1")
#     N_tx_var.grid(row=row, column=1); row+=1
#     tk.Label(frame_general, text="ant spacing (m)").grid(row=row, column=0)
#     d_spacing_var = tk.Entry(frame_general); d_spacing_var.insert(0, "0.1")
#     d_spacing_var.grid(row=row, column=1); row+=1
#
#     # --- Stage 1 ---
#     frame_stage1 = tk.LabelFrame(root, text="Stage 1 Parameters", padx=8, pady=8)
#     frame_stage1.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
#     row1=0
#     tk.Label(frame_stage1, text="SNR_basic_trainning").grid(row=row1, column=0)
#     SNR_basic_trainning_var = tk.Entry(frame_stage1); SNR_basic_trainning_var.insert(0, "50")
#     SNR_basic_trainning_var.grid(row=row1, column=1); row1+=1
#     tk.Label(frame_stage1, text="SNR_max").grid(row=row1, column=0)
#     SNR_max_var = tk.Entry(frame_stage1); SNR_max_var.insert(0, "30")
#     SNR_max_var.grid(row=row1, column=1); row1+=1
#     tk.Label(frame_stage1, text="max_iteration").grid(row=row1, column=0)
#     max_iteration_var = tk.Entry(frame_stage1); max_iteration_var.insert(0, "10")
#     max_iteration_var.grid(row=row1, column=1); row1+=1
#     tk.Label(frame_stage1, text="SNR_step").grid(row=row1, column=0)
#     SNR_step_var = tk.Entry(frame_stage1); SNR_step_var.insert(0, "1")
#     SNR_step_var.grid(row=row1, column=1); row1+=1
#     tk.Label(frame_stage1, text="BER_th").grid(row=row1, column=0)
#     BER_th_var = tk.Entry(frame_stage1); BER_th_var.insert(0, "1e-3")
#     BER_th_var.grid(row=row1, column=1); row1+=1
#
#     # --- Stage 2 ---
#     frame_stage2 = tk.LabelFrame(root, text="Stage 2 Parameters", padx=8, pady=8)
#     frame_stage2.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
#     row2=0
#     tk.Label(frame_stage2, text="epochs").grid(row=row2, column=0)
#     epochs_var = tk.Entry(frame_stage2); epochs_var.insert(0, "1")
#     epochs_var.grid(row=row2, column=1); row2+=1
#     tk.Label(frame_stage2, text="z0_type").grid(row=row2, column=0)
#     z0_type_var = ttk.Combobox(frame_stage2, values=["exp", "const", "linear"]); z0_type_var.set("exp")
#     z0_type_var.grid(row=row2, column=1); row2+=1
#     tk.Label(frame_stage2, text="z0_init").grid(row=row2, column=0)
#     z0_init_var = tk.Entry(frame_stage2); z0_init_var.insert(0, "0.1")
#     z0_init_var.grid(row=row2, column=1); row2+=1
#     tk.Label(frame_stage2, text="z0_end").grid(row=row2, column=0)
#     z0_end_var = tk.Entry(frame_stage2); z0_end_var.insert(0, "0.1")
#     z0_end_var.grid(row=row2, column=1); row2+=1
#     tk.Label(frame_stage2, text="B").grid(row=row2, column=0)
#     B_var = tk.Entry(frame_stage2); B_var.insert(0, "10")
#     B_var.grid(row=row2, column=1); row2+=1
#     tk.Label(frame_stage2, text="sub_stages (comma sep)").grid(row=row2, column=0)
#     sub_stages_var = tk.Entry(frame_stage2); sub_stages_var.insert(0, "0.4,0.5,0.1")
#     sub_stages_var.grid(row=row2, column=1); row2+=1
#
#     # Clone button
#     def clone_folder(update_dropdown):
#         src_folder = filedialog.askdirectory(initialdir=os.getcwd(), title="Select folder to clone")
#         if not src_folder: return
#         base_name = os.path.basename(src_folder)
#         new_name = simpledialog.askstring("Clone folder", f"Enter name for clone (default: {base_name}_clone):", initialvalue=base_name+"_clone")
#         if not new_name: return
#         dst_folder = os.path.join(os.getcwd(), new_name)
#         if os.path.exists(dst_folder): messagebox.showerror("Error", f"Destination '{dst_folder}' already exists."); return
#         try: shutil.copytree(src_folder, dst_folder); messagebox.showinfo("Success", f"Cloned to '{dst_folder}'"); update_dropdown()
#         except Exception as e: messagebox.showerror("Clone Error", str(e))
#     tk.Button(root, text="Clone Folder", command=lambda: clone_folder(update_folderlist)).grid(row=2, column=0, columnspan=3, pady=8, sticky="ew")
#
#     def submit():
#         try:
#             results['Name_of_model'] = name_var.get()
#             results['start_from_zero'] = bool(start_from_zero_var.get())
#             results['random_network'] = bool(random_network_var.get())
#             results['does_stage_1'] = bool(does_stage_1_var.get())
#             results['does_stage_2'] = bool(does_stage_2_var.get())
#             results['does_stage_3'] = bool(does_stage_3_var.get())
#             results['demod_type'] = demod_type_var.get()
#             results['N_transmitter'] = int(N_transmitter_var.get())
#             results['N_relays'] = int(N_relays_var.get())
#             results['N_users'] = [int(x) for x in N_users_var.get().split(',')]
#             results['N_channels'] = int(N_channels_var.get())
#             results['N_rx'] = int(N_rx_var.get())
#             results['N_tx'] = int(N_tx_var.get())
#             results['d_spacing'] = float(d_spacing_var.get())
#             results['SNR_basic_trainning'] = float(SNR_basic_trainning_var.get())
#             results['SNR_max'] = float(SNR_max_var.get())
#             results['max_iteration'] = int(max_iteration_var.get())
#             results['SNR_step'] = float(SNR_step_var.get())
#             results['BER_th'] = float(BER_th_var.get())
#             results['epochs'] = int(epochs_var.get())
#             results['z0_type'] = z0_type_var.get()
#             results['z0_init'] = float(z0_init_var.get())
#             results['z0_end'] = float(z0_end_var.get())
#             results['B'] = float(B_var.get())
#             results['sub_stages'] = [float(x) for x in sub_stages_var.get().split(',')]
#             root.quit(); root.destroy()
#         except Exception as e:
#             messagebox.showerror("Error", str(e))
#     tk.Button(root, text="Submit", command=submit).grid(row=3, column=0, columnspan=3, pady=8)
#     root.mainloop()
#     return results

import os, shutil
import customtkinter as ctk
from tkinter import filedialog, simpledialog, messagebox

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

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

def get_parameters_gui_grouped():
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