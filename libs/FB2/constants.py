from iso639 import languages as isoLanguages


FB2_LINK_PREFIX = "xlink"


def GetLanguages():
    if getattr(GetLanguages, "languages", None) is None:
        alpha1 = set(
            [getattr(language, "part1", None) for language in isoLanguages])
        alpha2b = set(
            [getattr(language, "part2b", None) for language in isoLanguages])
        alpha2t = set(
            [getattr(language, "part2t", None) for language in isoLanguages])
        alpha2 = set(
            list(alpha2b) + list(alpha2t - alpha2b))
        alpha3 = set(
            [getattr(language, "part3", None) for language in isoLanguages])
        alpha5 = set(
            [getattr(language, "part5", None) for language in isoLanguages])
        names = set(
            [getattr(language, "name", None) for language in isoLanguages])
        GetLanguages.languages = list(alpha1) + list(alpha2 - alpha1)
        GetLanguages.languages = (
            GetLanguages.languages
            + list(alpha3 - set(GetLanguages.languages)))
        GetLanguages.languages = (
            GetLanguages.languages
            + list(alpha5 - set(GetLanguages.languages)))
        GetLanguages.languages = (
            GetLanguages.languages
            + list(names - set(GetLanguages.languages)))
    return GetLanguages.languages
