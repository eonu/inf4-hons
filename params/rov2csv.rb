require 'csv'
require 'fileutils'

class ROV2CSV
  def initialize(rov, csv, translations: false, meta: false)
    @rov = rov
    @csv = csv.end_with?('.csv') ? csv : "#{csv}.csv"
    @meta = "#@csv.meta"

    puts "Exporting ROV data from #@rov to CSV..."

    prepare_csv(meta: meta)
    convert(translations: translations)

    puts "Finished converting #{rov}"
    puts "Exported CSV data to: #@csv"
    puts "Exported ROV metadata to: #@meta" if meta
    puts
  end

  private

  def prepare_csv(meta: false)
    # Create the directory if it doesn't exist
    path = File.dirname @csv
    FileUtils.mkdir_p path

    if meta
      # Create CSV metadata file
      @meta_file = File.new(@meta, "w")
    end
  end

  def convert(translations: false)
    CSV.open(@csv, "wb") do |csv|
      # Prepare header
      header = ["Rx", "Ry", "Rz"]
      header.concat(["Tx", "Ty", "Tz"]) if translations
      csv << header

      in_header = true
      File.open(@rov,'r').each do |line|
        line = line.strip

        # Enter the data section if we encounter 'END_OF_HEADER'
        in_header = false if line == 'END_OF_HEADER'

        if in_header
          # If still in header, add the line to the metadata file
          @meta_file.puts(line) if @meta_file
        else
          next if line == 'END_OF_HEADER'
          # No longer in header section
          r_x, r_y, r_z, t_x, t_y, t_z, * = line.split
          row = [r_x, r_y, r_z]
          row.concat([t_x, t_y, t_z]) if translations
          csv << row
        end
      end

      # Ensure the metadata file is closed
      @meta_file.close if @meta_file
    end
  end
end

if ARGV.include?('-m') && ARGV.include?('-t')
  args = ARGV
  args.delete '-m'
  args.delete '-t'
  ROV2CSV.new *(args), translations: true, meta: true
else
  if ARGV.include?('-m')
    # Include metadata
    args = ARGV
    args.delete '-m'
    ROV2CSV.new *(args), meta: true
  elsif ARGV.include?('-t')
    # Include translations
    args = ARGV
    args.delete '-t'
    ROV2CSV.new *(args), translations: true
  else
    ROV2CSV.new *ARGV
  end
end