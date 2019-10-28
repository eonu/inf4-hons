require 'fileutils'

SPEAKERS = %w[Adam Beve Bonn Bria Dani Ella Esmo Haze Iren Jack Liam Paul Soph]

def sort_formats
  formats = %w[axa dof.gz ea132 rov]
  FileUtils.mkdir formats
  formats.each do |fmt|
    FileUtils.mv Dir.glob("*.#{fmt}"), fmt
  end
end

SPEAKERS.each do |speaker|
  Dir.chdir(speaker) do
    system "zip -r #{speaker}.zip Head WS2 WS5"
    FileUtils.mv 'Head/Normalised', 'Normalized'
    FileUtils.rm_r %w[WS2 WS5]
    FileUtils.mv 'Head', 'Original'

    Dir.chdir("Original") { sort_formats }

    Dir.chdir("Normalized") do
      FileUtils.rm 'files_data.list', force: true
      FileUtils.rm_r 'Headerless'
      sort_formats
    end
  end
end