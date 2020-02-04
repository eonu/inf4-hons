require 'fileutils'

SPEAKERS = Dir.glob('*').select &File.method(:directory?)

SPEAKERS.each do |speaker|
  Dir.chdir(speaker) do
    name = speaker.downcase
    Dir.glob("**/#{speaker}*").each do |file|
      File.rename file, file.gsub(speaker, name)
    end
  end
end