import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Set the scan number
scan_number = 15

# Define base paths
base_path = r'C://Users//svens//OneDrive//Desktop//heruntergeladene_Daten_ESRF//scan15'
output_base_path = r'C://Users//svens//OneDrive//Desktop//heruntergeladene_Daten_ESRF//scan15'

# Construct file and directory paths using the scan number
filepath = os.path.join(base_path,'tio2_xpcs_trc.npz')
output_folder = os.path.join(output_base_path, f'scan{scan_number:04d}')

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load data
data = np.load(filepath, allow_pickle=True)
data_values = data['data']

ny, nx = data_values.shape
extent = [0, 52.33, 0, 52.33]
t1_max, t2_max = extent[1], extent[3]

highlight_color = '#ab1f20'

# Define the stretched exponential function for fitting
def stretched_exponential(x, tau_corr, alpha):
    return np.exp(-(x / tau_corr) ** alpha)

# Apply Gaussian smoothing to the data
smoothed_data = gaussian_filter(data_values, sigma=25)

# Modify the t_obs_values to include only five cuts
t_obs_values = [5.0, 7.5, 10.0, 12.5, 15.0]

# Colors for different t_obs values (now for 5 cuts)
t_obs_colors = ['black', 'black', 'black', 'black', 'black']

# Create a single figure for the 2-TCF with all five cuts
fig_2tcf = plt.figure(figsize=(10, 8))
ax_2tcf = fig_2tcf.add_subplot(111)

levels = np.linspace(np.min(smoothed_data), np.max(smoothed_data), 50)
contour = ax_2tcf.contourf(
    np.linspace(0, t1_max, nx),
    np.linspace(0, t2_max, ny),
    smoothed_data,
    levels=levels,
    cmap='turbo',
    vmin=1.05,
    vmax=1.285,
)

ax_2tcf.contour(
    np.linspace(0, t1_max, nx),
    np.linspace(0, t2_max, ny),
    smoothed_data,
    levels=levels[::2],
    colors='k',
    linewidths=0.5,
    alpha=0.3
)

cbar = plt.colorbar(contour, ax=ax_2tcf, pad=0.01)
cbar.set_label('Correlation', fontsize=18, labelpad=20)
cbar.ax.tick_params(labelsize=16)

# Draw diagonal line
ax_2tcf.plot([0, min(t1_max, t2_max)], [0, min(t1_max, t2_max)], 'k--', alpha=0.8, linewidth=1.5)

# 3. Make the CCS arrow 2 units longer (10 instead of 8)
arrow_length = 10

# For the largest t_obs, draw both arrows (CCS and ACS)
largest_t_obs = max(t_obs_values)
largest_t_obs_idx = t_obs_values.index(largest_t_obs)
largest_t_obs_color = t_obs_colors[largest_t_obs_idx]

# Draw all cuts and mark observation points
for idx, t_obs in enumerate(t_obs_values):
    color = t_obs_colors[idx]

    # Draw horizontal line from observation point to left axis
    ax_2tcf.plot([0, t_obs], [t_obs, t_obs],
                 color=color,
                 linestyle='--',
                 alpha=0.7,
                 linewidth=1.5)

    # Draw vertical line from observation point to bottom axis
    ax_2tcf.plot([t_obs, t_obs], [0, t_obs],
                 color=color,
                 linestyle='--',
                 alpha=0.7,
                 linewidth=1.5)

    # Mark the observation point
    ax_2tcf.plot(t_obs, t_obs, 'o',
                 color=color,
                 markersize=6,
                 markeredgecolor='white',
                 markeredgewidth=0.5,
                 label=f'$t_{{obs}} = {t_obs}$ s')

    # Only draw arrows for the largest t_obs
    if t_obs == largest_t_obs:
        # Draw CCS arrow (starting from t_obs, with fixed length)
        ax_2tcf.annotate('', xy=(t_obs + arrow_length, t_obs), xytext=(t_obs, t_obs),
                         arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax_2tcf.text(t_obs + 6, t_obs - 2, r'$\tau_{CCS}$', color='black', fontsize=18)

        # ACS arrow in opposite direction (1, -1)
        ax_2tcf.annotate('', xy=(t_obs + arrow_length / 1.41, t_obs - arrow_length / 1.41), xytext=(t_obs, t_obs),
                         arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax_2tcf.text(t_obs + 4, t_obs - arrow_length + 1.5, r'$\tau_{ACS}$', color='black', fontsize=18)

# Add titles and labels for 2-TCF
ax_2tcf.set_xlabel('$t_{1}$ [s]', fontsize=18)
ax_2tcf.set_ylabel('$t_{2}$ [s]', fontsize=18)
ax_2tcf.set_title('2-Time Correlation Function (2-TCF) with Multiple Cuts', fontsize=22, pad=20)
ax_2tcf.legend(loc='upper left', frameon=True)
ax_2tcf.legend(fontsize='18')

# Style the axes
ax_2tcf.spines['top'].set_visible(True)
ax_2tcf.spines['right'].set_visible(True)
ax_2tcf.spines['bottom'].set_visible(True)
ax_2tcf.spines['left'].set_visible(True)
ax_2tcf.tick_params(direction='in', length=6, width=1, colors='black', grid_alpha=0.5, labelsize=16)

# Save the 2-TCF figure
output_path = os.path.join(output_folder, 'all_cuts_2TCF')
plt.savefig(output_path + '.png', dpi=300, bbox_inches='tight')

# Define a function for time-window averaging
def time_window_average(tau_values, data_values, window_size=5):
    if len(tau_values) <= window_size:
        return tau_values, data_values

    # Sort the data by tau values
    sorted_indices = np.argsort(tau_values)
    tau_sorted = tau_values[sorted_indices]
    data_sorted = data_values[sorted_indices]

    # Initialize arrays for averaged values
    avg_tau = []
    avg_data = []

    # Create fewer output points for smoother curve
    step = max(1, len(tau_sorted) // 1000)

    # Perform rolling window average
    for i in range(0, len(tau_sorted), step):
        # Define window boundaries
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(tau_sorted), i + window_size // 2 + 1)

        # Average values in the window
        window_tau = tau_sorted[start_idx:end_idx]
        window_data = data_sorted[start_idx:end_idx]

        if len(window_tau) > 0:
            avg_tau.append(np.mean(window_tau))
            avg_data.append(np.mean(window_data))

    return np.array(avg_tau), np.array(avg_data)

# Now create two separate figures for the 1-TCF plots
# First figure for cuts 1-3
fig_1tcf_first = plt.figure(figsize=(15, 20))
gs_first = GridSpec(3, 2, figure=fig_1tcf_first, wspace=0.3, hspace=0.3)

# Second figure for cuts 4-5
fig_1tcf_second = plt.figure(figsize=(15, 13.33))  # Adjusted height to match the first figure
gs_second = GridSpec(2, 2, figure=fig_1tcf_second, wspace=0.3, hspace=0.3)

# Process all cuts
for row_idx, t_obs in enumerate(t_obs_values):
    color = t_obs_colors[row_idx]
    t_obs_y_idx = int(t_obs / t2_max * ny)
    t_obs_x_idx = int(t_obs / t1_max * nx)

    # Determine which figure and grid to use based on the cut index
    if row_idx < 3:  # First 3 cuts go to the first figure
        current_fig = fig_1tcf_first
        current_gs = gs_first
        current_row = row_idx
    else:  # Next 2 cuts go to the second figure
        current_fig = fig_1tcf_second
        current_gs = gs_second
        current_row = row_idx - 3  # Adjust row index for the second figure

    # CCS subplot (first column)
    ax_ccs = current_fig.add_subplot(current_gs[current_row, 0])
    ax_ccs.set_xscale('log')
    # Set ylim after plotting data to accommodate unnormalized range
    ax_ccs.set_xlim([0.1, t1_max])
    ax_ccs.set_xlabel(r'$\tau$ [s]', fontsize=18)
    ax_ccs.set_ylabel('Correlation', fontsize=18)
    ax_ccs.set_title(f'CCS at $t_{{obs}} = {t_obs}$ s', fontsize=18)

    # Process CCS cut
    ccs_cut = []
    tau_values_ccs = []

    if t_obs_y_idx < ny and t_obs_x_idx < nx:
        for j in range(t_obs_x_idx, nx):
            tau = (j - t_obs_x_idx) * t1_max / nx
            tau_values_ccs.append(tau)
            ccs_cut.append(data_values[t_obs_y_idx, j])

        if len(ccs_cut) > 0:
            ccs_cut = np.array(ccs_cut)
            tau_values_ccs = np.array(tau_values_ccs)

            tau_values_ccs = np.maximum(tau_values_ccs, 0.01)

            # Apply time-window averaging to improve statistics
            tau_values_ccs_avg, ccs_cut_avg = time_window_average(tau_values_ccs, ccs_cut, window_size=5)

            # NO NORMALIZATION - using raw data values

            ax_ccs.plot(tau_values_ccs_avg, ccs_cut_avg,
                        color=color,
                        linestyle=' ',
                        marker='x',
                        markersize=5,
                        markevery=1,
                        alpha=0.8,
                        label=f'CCS Data')

            fit_mask = (tau_values_ccs_avg >= 1) & (tau_values_ccs_avg <= 50)
            if np.sum(fit_mask) > 3:
                try:
                    # Adjust bounds and initial parameters for the fit
                    # to match the unnormalized data range
                    data_min = np.min(ccs_cut_avg[fit_mask])
                    data_max = np.max(ccs_cut_avg[fit_mask])

                    # Define a modified stretched exponential function that handles unnormalized data
                    def mod_stretched_exponential(t, tau, beta, amplitude, offset):
                        return amplitude * np.exp(-(t / tau) ** beta) + offset

                    popt, pcov = curve_fit(mod_stretched_exponential,
                                           tau_values_ccs_avg[fit_mask],
                                           ccs_cut_avg[fit_mask],
                                           p0=[10, 1.3, data_max - data_min, data_min],
                                           bounds=([0.1, 0.1, 0, -np.inf], [50, 3, np.inf, np.inf]))

                    tau_corr_ccs, alpha_ccs, amplitude_ccs, offset_ccs = popt
                    perr = np.sqrt(np.diag(pcov))

                    tau_fit = np.logspace(-1, np.log10(t1_max), 50)
                    fit_curve = mod_stretched_exponential(tau_fit, tau_corr_ccs, alpha_ccs, amplitude_ccs, offset_ccs)

                    # Plot CCS fit
                    ax_ccs.plot(tau_fit, fit_curve, color=highlight_color, linestyle='-', linewidth=1.5, alpha=0.7, label=f'Fit')

                    ax_ccs.text(0.05, 0.35,
                                r'$\beta^{CCS}=%.2f \pm %.2f$' % (alpha_ccs, perr[1]) + '\n' + r'$\tau_{CCS}=%.2f \pm %.2f$' % (tau_corr_ccs, perr[0]),
                                transform=ax_ccs.transAxes, fontsize=18, color=highlight_color, alpha=0.99)

                    # Calculate R² for the CCS fit
                    ccs_y_pred = mod_stretched_exponential(tau_values_ccs_avg[fit_mask], tau_corr_ccs, alpha_ccs, amplitude_ccs, offset_ccs)
                    ccs_r2 = 1 - np.sum((ccs_cut_avg[fit_mask] - ccs_y_pred)**2) / np.sum((ccs_cut_avg[fit_mask] - np.mean(ccs_cut_avg[fit_mask]))**2)
                    ax_ccs.text(0.05, 0.25, f"R² = {ccs_r2:.3f}", transform=ax_ccs.transAxes, fontsize=18, color=highlight_color, alpha=0.99)

                except Exception as e:
                    print(f"Fitting failed for CCS cut at t_obs={t_obs}: {e}")

            # Set y-limits after plotting to accommodate data range
            y_data_min = np.min(ccs_cut_avg)
            y_data_max = np.max(ccs_cut_avg)
            y_margin = (y_data_max - y_data_min) * 0.1  # 10% margin
            ax_ccs.set_ylim([y_data_min - y_margin, y_data_max + y_margin])

    # ACS subplot (second column)
    ax_acs = current_fig.add_subplot(current_gs[current_row, 1])
    ax_acs.set_xscale('log')
    # Set ylim after plotting data to accommodate unnormalized range
    ax_acs.set_xlim([0.1, t1_max])
    ax_acs.set_xlabel(r'$\tau$ [s]', fontsize=18)
    ax_acs.set_ylabel('Correlation', fontsize=18)
    ax_acs.set_title(f'ACS at $t_{{obs}} = {t_obs}$ s', fontsize=18)

    # Process ACS cut - in direction (1, -1)
    acs_cut = []
    tau_values_acs = []

    if t_obs_y_idx < ny and t_obs_x_idx < nx:
        max_steps = min(nx - t_obs_x_idx - 1, t_obs_y_idx)  # Maximum steps 

        for step in range(max_steps + 1):
            x_idx = t_obs_x_idx + step
            y_idx = t_obs_y_idx - step

            if x_idx < nx and y_idx >= 0:
                # Calculate tau as Euclidean distance from diagonal
                tau = step * np.sqrt(2) * t1_max / nx  # Multiply by sqrt(2) ---> moving diagonally
                tau_values_acs.append(tau)
                acs_cut.append(data_values[y_idx, x_idx])

        if len(acs_cut) > 0:
            acs_cut = np.array(acs_cut)
            tau_values_acs = np.array(tau_values_acs)

            tau_values_acs = np.maximum(tau_values_acs, 0.01)

            # Apply time-window averaging to improve statistics
            tau_values_acs_avg, acs_cut_avg = time_window_average(tau_values_acs, acs_cut, window_size=5)

            # NO NORMALIZATION - using raw data values

            ax_acs.plot(tau_values_acs_avg, acs_cut_avg,
                        color=color,
                        linestyle=' ',
                        marker='o',
                        markersize=5,
                        markevery=1,
                        alpha=0.8,
                        label=f'ACS Data')

            fit_mask = (tau_values_acs_avg >= 1) & (tau_values_acs_avg <= 50)
            if np.sum(fit_mask) > 3:
                try:
                    # For the fit, adjust the bounds and initial parameters to match the unnormalized data range
                    data_min = np.min(acs_cut_avg[fit_mask])
                    data_max = np.max(acs_cut_avg[fit_mask])

                    # Define a modified stretched exponential function that handles unnormalized data
                    def mod_stretched_exponential(t, tau, beta, amplitude, offset):
                        return amplitude * np.exp(-(t / tau) ** beta) + offset

                    popt, pcov = curve_fit(mod_stretched_exponential,
                                           tau_values_acs_avg[fit_mask],
                                           acs_cut_avg[fit_mask],
                                           p0=[10, 1.8, data_max - data_min, data_min],
                                           bounds=([0.1, 0.1, 0, -np.inf], [50, 3, np.inf, np.inf]))

                    tau_corr_acs, alpha_acs, amplitude_acs, offset_acs = popt
                    perr = np.sqrt(np.diag(pcov))

                    tau_fit = np.logspace(-1, np.log10(t1_max), 50)
                    fit_curve = mod_stretched_exponential(tau_fit, tau_corr_acs, alpha_acs, amplitude_acs, offset_acs)

                    # Plot ACS fit
                    ax_acs.plot(tau_fit, fit_curve, color=highlight_color, linestyle='-', linewidth=1.5, alpha=0.7, label=f'Fit')

                    ax_acs.text(0.05, 0.35,
                                r'$\beta^{ACS}=%.2f \pm %.2f$' % (alpha_acs, perr[1]) + '\n' + r'$\tau_{ACS}=%.2f \pm %.2f$' % (tau_corr_acs, perr[0]),
                                transform=ax_acs.transAxes, fontsize=18, color=highlight_color, alpha=0.99)

                    # Calculate R² for the ACS fit
                    acs_y_pred = mod_stretched_exponential(tau_values_acs_avg[fit_mask], tau_corr_acs, alpha_acs, amplitude_acs, offset_acs)
                    acs_r2 = 1 - np.sum((acs_cut_avg[fit_mask] - acs_y_pred)**2) / np.sum((acs_cut_avg[fit_mask] - np.mean(acs_cut_avg[fit_mask]))**2)
                    ax_acs.text(0.05, 0.25, f"R² = {acs_r2:.3f}", transform=ax_acs.transAxes, fontsize=18, color=highlight_color, alpha=0.99)

                except Exception as e:
                    print(f"Fitting failed for ACS cut at t_obs={t_obs}: {e}")

            # Set y-limits after plotting to accommodate data range
            y_data_min = np.min(acs_cut_avg)
            y_data_max = np.max(acs_cut_avg)
            y_margin = (y_data_max - y_data_min) * 0.1  # 10 percent margin
            ax_acs.set_ylim([y_data_min - y_margin, y_data_max + y_margin])

    # Style the axes
    for ax in [ax_ccs, ax_acs]:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.tick_params(direction='in', length=6, width=1, colors='black', grid_alpha=0.5, pad=7.5, labelsize=16)
        ax.legend(loc='lower left', frameon=True, framealpha=1.0, fontsize=16)

# Add main titles for both 1-TCF figures
fig_1tcf_first.suptitle('1-Time Correlation Functions (1-TCF) for Observation Times 5-10s', fontsize=22, y=0.925)
fig_1tcf_second.suptitle('1-Time Correlation Functions (1-TCF) for Observation Times 12.5-15s', fontsize=22, y=0.95)

# Save both 1-TCF figures
output_path_first = os.path.join(output_folder, 'cuts_1-3_1TCF')
plt.figure(fig_1tcf_first.number)
plt.savefig(output_path_first + '.png', dpi=300, bbox_inches='tight')

output_path_second = os.path.join(output_folder, 'cuts_4-5_1TCF')
plt.figure(fig_1tcf_second.number)
plt.savefig(output_path_second + '.png', dpi=300, bbox_inches='tight')

# New section for additional cuts and plots
# Define new t_obs_values for every second from 1 to 25
new_t_obs_values = np.arange(1, 26)

# Initialize lists to store alpha and tau values for CCS and ACS
alpha_ccs_values = []
alpha_acs_values = []
tau_corr_ccs_values = []
tau_corr_acs_values = []
tau_corr_ccs_errors = []
tau_corr_acs_errors = []
alpha_ccs_errors = []
alpha_acs_errors = []

# Process all new cuts
for t_obs in new_t_obs_values:
    t_obs_y_idx = int(t_obs / t2_max * ny)
    t_obs_x_idx = int(t_obs / t1_max * nx)

    # CCS cut
    if t_obs_y_idx < ny and t_obs_x_idx < nx:
        ccs_cut = []
        tau_values_ccs = []
        for j in range(t_obs_x_idx, nx):
            tau = (j - t_obs_x_idx) * t1_max / nx
            tau_values_ccs.append(tau)
            ccs_cut.append(data_values[t_obs_y_idx, j])

        if len(ccs_cut) > 0:
            ccs_cut = np.array(ccs_cut)
            tau_values_ccs = np.array(tau_values_ccs)
            tau_values_ccs = np.maximum(tau_values_ccs, 0.01)
            tau_values_ccs_avg, ccs_cut_avg = time_window_average(tau_values_ccs, ccs_cut, window_size=5)

            # NO NORMALIZATION for fitting

            fit_mask = (tau_values_ccs_avg >= 1) & (tau_values_ccs_avg <= 50)
            if np.sum(fit_mask) > 3:
                try:
                    # Get data range for unnormalized fit
                    data_min = np.min(ccs_cut_avg[fit_mask])
                    data_max = np.max(ccs_cut_avg[fit_mask])

                    # Use modified stretched exponential with amplitude and offset
                    def mod_stretched_exponential(t, tau, beta, amplitude, offset):
                        return amplitude * np.exp(-(t / tau) ** beta) + offset

                    popt, pcov = curve_fit(mod_stretched_exponential,
                                           tau_values_ccs_avg[fit_mask],
                                           ccs_cut_avg[fit_mask],
                                           p0=[10, 1.3, data_max - data_min, data_min],
                                           bounds=([0.1, 0.1, 0, -np.inf], [50, 3, np.inf, np.inf]))

                    tau_corr_ccs, alpha_ccs, _, _ = popt
                    perr = np.sqrt(np.diag(pcov))
                    tau_corr_ccs_errors.append(perr[0])
                    alpha_ccs_errors.append(perr[1])
                    tau_corr_ccs_values.append(tau_corr_ccs)
                    alpha_ccs_values.append(alpha_ccs)
                except Exception as e:
                    print(f"Fitting failed for CCS cut at t_obs={t_obs}: {e}")

    # ACS cut
    if t_obs_y_idx < ny and t_obs_x_idx < nx:
        acs_cut = []
        tau_values_acs = []
        max_steps = min(nx - t_obs_x_idx - 1, t_obs_y_idx)

        for step in range(max_steps + 1):
            x_idx = t_obs_x_idx + step
            y_idx = t_obs_y_idx - step

            if x_idx < nx and y_idx >= 0:
                tau = step * np.sqrt(2) * t1_max / nx
                tau_values_acs.append(tau)
                acs_cut.append(data_values[y_idx, x_idx])

        if len(acs_cut) > 0:
            acs_cut = np.array(acs_cut)
            tau_values_acs = np.array(tau_values_acs)
            tau_values_acs = np.maximum(tau_values_acs, 0.01)
            tau_values_acs_avg, acs_cut_avg = time_window_average(tau_values_acs, acs_cut, window_size=5)

            # NO NORMALIZATION for fitting

            fit_mask = (tau_values_acs_avg >= 1) & (tau_values_acs_avg <= 100)
            if np.sum(fit_mask) > 3:
                try:
                    # Get data range for unnormalized fit
                    data_min = np.min(acs_cut_avg[fit_mask])
                    data_max = np.max(acs_cut_avg[fit_mask])

                    # Use modified stretched exponential with amplitude and offset
                    def mod_stretched_exponential(t, tau, beta, amplitude, offset):
                        return amplitude * np.exp(-(t / tau) ** beta) + offset

                    popt, pcov = curve_fit(mod_stretched_exponential,
                                           tau_values_acs_avg[fit_mask],
                                           acs_cut_avg[fit_mask],
                                           p0=[10, 1.8, data_max - data_min, data_min],
                                           bounds=([0.1, 0.1, 0, -np.inf], [100, 3, np.inf, np.inf]))

                    tau_corr_acs, alpha_acs, _, _ = popt
                    perr = np.sqrt(np.diag(pcov))
                    tau_corr_acs_errors.append(perr[0])
                    alpha_acs_errors.append(perr[1])
                    tau_corr_acs_values.append(tau_corr_acs)
                    alpha_acs_values.append(alpha_acs)
                except Exception as e:
                    print(f"Fitting failed for ACS cut at t_obs={t_obs}: {e}")

# Convert lists to numpy arrays for easier manipulation
tau_corr_ccs_values = np.array(tau_corr_ccs_values)
tau_corr_acs_values = np.array(tau_corr_acs_values)
alpha_ccs_values = np.array(alpha_ccs_values)
alpha_acs_values = np.array(alpha_acs_values)
tau_corr_ccs_errors = np.array(tau_corr_ccs_errors)
tau_corr_acs_errors = np.array(tau_corr_acs_errors)
alpha_ccs_errors = np.array(alpha_ccs_errors)
alpha_acs_errors = np.array(alpha_acs_errors)
t_obs_ccs = np.array(new_t_obs_values[:len(tau_corr_ccs_values)])
t_obs_acs = np.array(new_t_obs_values[:len(tau_corr_acs_values)])

# Function to identify and remove outliers using z-score based on a specific fit
def remove_outliers_with_fit(x_data, y_data, fit_params=None, threshold=1):
    if len(x_data) < 3:
        return x_data, y_data, np.ones(len(x_data), dtype=bool)

    # Use provided fit parameters or calculate them
    if fit_params is None:
        fit_params = np.polyfit(x_data, y_data, 1)
    
    # Calculate predicted values based on the fit
    y_pred = np.polyval(fit_params, x_data)

    # Calculate residuals and z-scores
    residuals = y_data - y_pred
    z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))

    # Create mask for non-outliers
    valid_mask = z_scores < threshold

    return x_data[valid_mask], y_data[valid_mask], valid_mask

# Original remove_outliers function as a fallback
def remove_outliers(x_data, y_data, threshold=1):
    return remove_outliers_with_fit(x_data, y_data, None, threshold)

# Function to format equation string
def format_equation(params):
    a, b = params
    sign = "+" if b >= 0 else ""
    return f"y = {a:.2f}x{sign}{b:.2f}"

# Function to calculate R² for any fit
def calculate_r2(y_actual, y_predicted):
    ss_total = np.sum((y_actual - np.mean(y_actual))**2)
    ss_residual = np.sum((y_actual - y_predicted)**2)
    return 1 - (ss_residual / ss_total)

# Aging Plot (with 2 subplots: ACS and CCS)
fig_aging = plt.figure(figsize=(15, 7))
gs_aging = GridSpec(1, 2, figure=fig_aging, wspace=0.3, hspace=0.3)

# CCS Aging Subplot
ax_aging_ccs = fig_aging.add_subplot(gs_aging[0, 0])

# Plot all CCS data points without error bars
ax_aging_ccs.plot(t_obs_ccs, tau_corr_ccs_values, 'x', color='gray', alpha=0.7, label='_nolegend_')

# Remove outliers and get clean data for CCS
t_obs_ccs_clean, tau_ccs_clean, valid_ccs_mask = remove_outliers(t_obs_ccs, tau_corr_ccs_values, threshold=0.75)

# Make sure error arrays have the right lengths for CCS
tau_ccs_errors_filtered = np.array(tau_corr_ccs_errors[:len(valid_ccs_mask)])[valid_ccs_mask]

# Plot CCS data points with error bars - less restrictive filtering
if len(tau_ccs_errors_filtered) > 0:
    # Use more permissive threshold or no filtering
    median_err = np.median(tau_ccs_errors_filtered[tau_ccs_errors_filtered > 0])
    err_threshold = 5 * median_err  # More permissive threshold
    
    # Create mask for reasonably sized error bars
    mask = tau_ccs_errors_filtered <= err_threshold
    
    # Plot all points with capsize added
    ax_aging_ccs.errorbar(t_obs_ccs_clean, tau_ccs_clean, 
                    yerr=np.where(mask, tau_ccs_errors_filtered, 0),  # Zero error for outliers
                    fmt='x', color='black', label=r'$\tau_{CCS}$', capsize=3)
else:
    ax_aging_ccs.plot(t_obs_ccs_clean, tau_ccs_clean, 'x', color='black', label=r'$\tau_{CCS}$')

# Perform linear fits on clean CCS data
ccs_fit = np.polyfit(t_obs_ccs_clean, tau_ccs_clean, 1)

# Calculate R² for the CCS fit
ccs_y_pred = np.polyval(ccs_fit, t_obs_ccs_clean)
ccs_r2 = calculate_r2(tau_ccs_clean, ccs_y_pred)

# Format fit equation for CCS
ccs_eq = format_equation(ccs_fit)

# Plot linear fit for CCS
ax_aging_ccs.plot(new_t_obs_values, np.polyval(ccs_fit, new_t_obs_values), color=highlight_color, linestyle='--', label=r'$\tau_{CCS}$ Fit')

# Create custom text box for CCS fit data
ccs_text = f'CCS: {ccs_eq}\nR² = {ccs_r2:.3f}'

# Add custom text box to the CCS plot
ax_aging_ccs.text(0.02, 0.735, ccs_text, transform=ax_aging_ccs.transAxes, fontsize=16, color=highlight_color, alpha=0.99,
              bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.5'))

ax_aging_ccs.set_xlabel('Observation Time $t_{obs}$ [s]', fontsize=18)
ax_aging_ccs.set_ylabel('Relaxation Time $\\tau$ [s]', fontsize=18)
ax_aging_ccs.set_title('CCS Aging Plot', fontsize=22)
ax_aging_ccs.legend(loc='upper left', frameon=True, fontsize=16)

# Style the axes for CCS
ax_aging_ccs.spines['top'].set_visible(True)
ax_aging_ccs.spines['right'].set_visible(True)
ax_aging_ccs.spines['bottom'].set_visible(True)
ax_aging_ccs.spines['left'].set_visible(True)
ax_aging_ccs.tick_params(direction='in', length=6, width=1, colors='black', grid_alpha=0.5, labelsize=16)

# ACS Aging Subplot
ax_aging_acs = fig_aging.add_subplot(gs_aging[0, 1])

# Plot all ACS data points without error bars
ax_aging_acs.plot(t_obs_acs, tau_corr_acs_values, 'o', color='gray', alpha=0.7, label='_nolegend_')

# Remove outliers and get clean data for ACS
t_obs_acs_clean, tau_acs_clean, valid_acs_mask = remove_outliers(t_obs_acs, tau_corr_acs_values, threshold=0.75)

# Make sure error arrays have the right lengths for ACS
tau_acs_errors_filtered = np.array(tau_corr_acs_errors[:len(valid_acs_mask)])[valid_acs_mask]

# Plot ACS data points with error bars
if len(tau_acs_errors_filtered) > 0:
    median_err = np.median(tau_acs_errors_filtered[tau_acs_errors_filtered > 0])
    err_threshold = 5 * median_err
    mask = tau_acs_errors_filtered <= err_threshold
    
    ax_aging_acs.errorbar(t_obs_acs_clean, tau_acs_clean,
                    yerr=np.where(mask, tau_acs_errors_filtered, 0),
                    fmt='o', color='black', label=r'$\tau_{ACS}$', capsize=3)
else:
    ax_aging_acs.plot(t_obs_acs_clean, tau_acs_clean, 'o', color='black', label=r'$\tau_{ACS}$')

# Perform linear fits on clean ACS data
acs_fit = np.polyfit(t_obs_acs_clean, tau_acs_clean, 1)

# Calculate R² for ACS fit
acs_y_pred = np.polyval(acs_fit, t_obs_acs_clean)
acs_r2 = calculate_r2(tau_acs_clean, acs_y_pred)

# Format fit equation for ACS
acs_eq = format_equation(acs_fit)

# Plot linear fit for ACS
ax_aging_acs.plot(new_t_obs_values, np.polyval(acs_fit, new_t_obs_values), color=highlight_color, linestyle='--', label=r'$\tau_{ACS}$ Fit')

# Create custom text box for ACS fit data
acs_text = f'ACS: {acs_eq}\nR² = {acs_r2:.3f}'

# Add custom text box to the ACS plot
ax_aging_acs.text(0.02, 0.735, acs_text, transform=ax_aging_acs.transAxes, fontsize=16, color=highlight_color, alpha=0.99,
              bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.5'))

ax_aging_acs.set_xlabel('Observation Time $t_{obs}$ [s]', fontsize=18)
ax_aging_acs.set_ylabel('Relaxation Time $\\tau$ [s]', fontsize=18)
ax_aging_acs.set_title('ACS Aging Plot', fontsize=22)
ax_aging_acs.legend(loc='upper left', frameon=True, fontsize=16)

# Style the axes for ACS
ax_aging_acs.spines['top'].set_visible(True)
ax_aging_acs.spines['right'].set_visible(True)
ax_aging_acs.spines['bottom'].set_visible(True)
ax_aging_acs.spines['left'].set_visible(True)
ax_aging_acs.tick_params(direction='in', length=6, width=1, colors='black', grid_alpha=0.5, labelsize=16)

# Save the aging plot
output_path_aging = os.path.join(output_folder, 'separate_aging_plots')
plt.savefig(output_path_aging + '.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

# Exponent Analysis Plot (with 2 subplots: ACS and CCS)
fig_exponent = plt.figure(figsize=(15, 7))
gs_exponent = GridSpec(1, 2, figure=fig_exponent, wspace=0.3, hspace=0.3)

# CCS Exponent Analysis Subplot
ax_exponent_ccs = fig_exponent.add_subplot(gs_exponent[0, 0])

# Convert alpha values to numpy arrays if needed
alpha_ccs_array = np.array(alpha_ccs_values)
t_obs_alpha_ccs = np.array(new_t_obs_values[:len(alpha_ccs_array)])

# Plot all CCS data points without error bars
ax_exponent_ccs.plot(t_obs_alpha_ccs, alpha_ccs_array, 'x', color='gray', alpha=0.7, label='_nolegend_')

# Manual linear fit parameters for CCS exponent (KWW)
ccs_alpha_manual_slope = -0.02  # Educated guess
ccs_alpha_manual_intercept = 1.135  # Educated guess
ccs_alpha_fit = np.array([ccs_alpha_manual_slope, ccs_alpha_manual_intercept])

# Remove outliers based on the manual fit for CCS
t_obs_alpha_ccs_clean, alpha_ccs_clean, valid_alpha_ccs_mask = remove_outliers_with_fit(
    t_obs_alpha_ccs, alpha_ccs_array, ccs_alpha_fit)

# Make sure error arrays have the right lengths
alpha_ccs_errors_filtered = np.array(alpha_ccs_errors[:len(valid_alpha_ccs_mask)])[valid_alpha_ccs_mask]

# Plot CCS data points with error bars
if len(alpha_ccs_errors_filtered) > 0:
    median_err = np.median(alpha_ccs_errors_filtered[alpha_ccs_errors_filtered > 0])
    err_threshold = 5 * median_err
    mask = alpha_ccs_errors_filtered <= err_threshold
    
    ax_exponent_ccs.errorbar(t_obs_alpha_ccs_clean, alpha_ccs_clean,
                      yerr=np.where(mask, alpha_ccs_errors_filtered, 0),
                      fmt='x', color='black', label=r'$\beta^{CCS}$', capsize=3)
else:
    ax_exponent_ccs.plot(t_obs_alpha_ccs_clean, alpha_ccs_clean, 'x', color='black', label=r'$\beta^{CCS}$')

# Calculate R² for the manual CCS fit using proper predicted values
ccs_alpha_y_pred = np.polyval(ccs_alpha_fit, t_obs_alpha_ccs_clean)
ccs_alpha_r2 = calculate_r2(alpha_ccs_clean, ccs_alpha_y_pred)

# Format fit equation for CCS
ccs_alpha_eq = format_equation(ccs_alpha_fit)

# Plot linear fit for CCS
ax_exponent_ccs.plot(new_t_obs_values, np.polyval(ccs_alpha_fit, new_t_obs_values), color=highlight_color, linestyle='--', label=r'$\beta^{CCS}$ Fit (Manual)')

# Create custom text box for CCS fit data
ccs_alpha_text = f'CCS: {ccs_alpha_eq}\nR² = {ccs_alpha_r2:.3f} (Manual Fit)'

# Add custom text box to the CCS plot
ax_exponent_ccs.text(0.025, 0.03, ccs_alpha_text, transform=ax_exponent_ccs.transAxes, fontsize=16, color=highlight_color, alpha=0.99,
              bbox=dict(facecolor='white', alpha=0.99, edgecolor='none', boxstyle='round,pad=0.5'))

ax_exponent_ccs.set_xlabel('Observation Time $t_{obs}$ [s]', fontsize=18)
ax_exponent_ccs.set_ylabel('Exponent $\\beta$', fontsize=18)
ax_exponent_ccs.set_title('CCS KWW-Exponent Analysis', fontsize=22)
ax_exponent_ccs.legend(loc='upper right', frameon=True, fontsize=16)

# Style the axes for CCS
ax_exponent_ccs.spines['top'].set_visible(True)
ax_exponent_ccs.spines['right'].set_visible(True)
ax_exponent_ccs.spines['bottom'].set_visible(True)
ax_exponent_ccs.spines['left'].set_visible(True)
ax_exponent_ccs.tick_params(direction='in', length=6, width=1, colors='black', grid_alpha=0.5, labelsize=16)

# ACS Exponent Analysis Subplot
ax_exponent_acs = fig_exponent.add_subplot(gs_exponent[0, 1])

# Convert alpha values to numpy arrays if needed
alpha_acs_array = np.array(alpha_acs_values)
t_obs_alpha_acs = np.array(new_t_obs_values[:len(alpha_acs_array)])

# Plot all ACS data points without error bars
ax_exponent_acs.plot(t_obs_alpha_acs, alpha_acs_array, 'o', color='gray', alpha=0.7, label='_nolegend_')

# Remove outliers for ACS
t_obs_alpha_acs_clean, alpha_acs_clean, valid_alpha_acs_mask = remove_outliers(
    t_obs_alpha_acs, alpha_acs_array)

# Make sure error arrays have the right lengths
alpha_acs_errors_filtered = np.array(alpha_acs_errors[:len(valid_alpha_acs_mask)])[valid_alpha_acs_mask]

# Plot ACS data points with error bars
if len(alpha_acs_errors_filtered) > 0:
    median_err = np.median(alpha_acs_errors_filtered[alpha_acs_errors_filtered > 0])
    err_threshold = 5 * median_err
    mask = alpha_acs_errors_filtered <= err_threshold
    
    ax_exponent_acs.errorbar(t_obs_alpha_acs_clean, alpha_acs_clean,
                      yerr=np.where(mask, alpha_acs_errors_filtered, 0),
                      fmt='o', color='black', label=r'$\beta^{ACS}$', capsize=3)
else:
    ax_exponent_acs.plot(t_obs_alpha_acs_clean, alpha_acs_clean, 'o', color='black', label=r'$\beta^{ACS}$')
    
# For ACS, perform automatic fit
acs_alpha_fit = np.polyfit(t_obs_alpha_acs_clean, alpha_acs_clean, 1)

# Calculate R² for the automatic ACS fit
acs_alpha_y_pred = np.polyval(acs_alpha_fit, t_obs_alpha_acs_clean)
acs_alpha_r2 = calculate_r2(alpha_acs_clean, acs_alpha_y_pred)

# Format fit equation for ACS
acs_alpha_eq = format_equation(acs_alpha_fit)

# Plot linear fit for ACS
ax_exponent_acs.plot(new_t_obs_values, np.polyval(acs_alpha_fit, new_t_obs_values), color=highlight_color, linestyle='--', label=r'$\beta^{ACS}$ Fit')

# Create custom text box for ACS fit data
acs_alpha_text = f'ACS: {acs_alpha_eq}\nR² = {acs_alpha_r2:.3f}'

# Add custom text box to the ACS plot
ax_exponent_acs.text(0.025, 0.03, acs_alpha_text, transform=ax_exponent_acs.transAxes, fontsize=16, color=highlight_color, alpha=0.99,
              bbox=dict(facecolor='white', alpha=0.99, edgecolor='none', boxstyle='round,pad=0.5'))

ax_exponent_acs.set_xlabel('Observation Time $t_{obs}$ [s]', fontsize=18)
ax_exponent_acs.set_ylabel('Exponent $\\beta$', fontsize=18)
ax_exponent_acs.set_title('ACS KWW-Exponent Analysis', fontsize=22)
ax_exponent_acs.legend(loc='upper right', frameon=True, fontsize=16)

# Style the axes for ACS
ax_exponent_acs.spines['top'].set_visible(True)
ax_exponent_acs.spines['right'].set_visible(True)
ax_exponent_acs.spines['bottom'].set_visible(True)
ax_exponent_acs.spines['left'].set_visible(True)
ax_exponent_acs.tick_params(direction='in', length=6, width=1, colors='black', grid_alpha=0.5, labelsize=16)

# Save the exponent analysis plot
output_path_exponent = os.path.join(output_folder, 'separate_exponent_analysis_plots')
plt.savefig(output_path_exponent + '.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

